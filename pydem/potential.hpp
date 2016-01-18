#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cassert>

#include <vexcl/vexcl.hpp>

inline void precondition(bool cond, std::string error_message) {
    if (!cond) throw std::runtime_error(error_message);
}

class potential {
    public:
        potential(
                std::vector<double> x,
                std::vector<double> y,
                double xmin   = 0,
                double xmax   = 5,
                int grid_size = 8
                )
            : hinv(1), phi(x.empty() ? 0 : grid_size, 0.0)
        {
            precondition(x.size() == y.size(), "x and y arrays differ in size");

            if (x.size()) {
                hinv = (grid_size - 3) / (xmax - xmin);
                x0 = xmin - 1.0 / hinv;

                std::vector<double> delta(grid_size, 0.0);
                std::vector<double> omega(grid_size, 0.0);

                for(unsigned k = 0; k < x.size(); ++k) {
                    if (x[k] < xmin || x[k] > xmax) continue;

                    double u = (x[k] - x0) * hinv;
                    int    i = static_cast<int>(u) - 1;
                    double s = u - static_cast<int>(u);

                    double w[4] = {
                        Bspline(0, s),
                        Bspline(1, s),
                        Bspline(2, s),
                        Bspline(3, s)
                    };
                    double sum_w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3];

                    for(int j = 0; j < 4; ++j) {
                        double w1 = w[j];
                        double w2 = w1 * w1;
                        double phi = y[k] * w1 / sum_w2;

                        delta[i + j] += w2 * phi;
                        omega[i + j] += w2;
                    }
                }

                for(int i = 0; i < grid_size; ++i) {
                    phi[i] = (omega[i] == 0.0 ? 0.0 : delta[i] / omega[i]);
                }
            }
        }

        virtual double approx(double) const = 0;
        virtual void def_approx(vex::backend::source_generator &src, std::string name) const = 0;

        double operator()(double x) const {
            double y = approx(x);

            if (!phi.empty()) {
                double u = (x - x0) * hinv;
                int    i = floor(u) - 1;
                double s = u - floor(u);

                for(int k = i, j = 0; j < 4; ++j, ++k) {
                    if (k >= 0 && k < static_cast<int>(phi.size()))
                        y += Bspline(j, s) * phi[k];
                }
            }

            return y;
        }

        void def_potential(vex::backend::source_generator &src, std::string name) const {
            def_approx(src, name);

            src << std::scientific << std::setprecision(12);

            src.function<cl_double2>(name).open("(")
                .parameter<cl_double2>("d")
                .close(")").open("{");
            src.new_line() << "double r = length(d);";
            src.new_line() << "double y = " << name << "_approx(r);";

            if (!phi.empty()) {
                src.new_line() << "const double phi[] = ";
                src.open("{");
                src.new_line() << phi[0];
                for(size_t i = 1; i < phi.size(); ++i)
                    src.new_line() << ", " << phi[i];
                src.close("};");

                src.new_line() << "double u = (r - (" << x0 << ")) * (" << hinv << ");";
                src.new_line() << "int    i = (int)u - 1;";
                src.new_line() << "double s = u - (int)u;";
                src.new_line() << "for(int k = i, j = 0; j < 4; ++j, ++k)";
                src.new_line() << "    if (k >= 0 && k < " << phi.size() << ")";
                src.new_line() << "        y += Bspline(j, s) * phi[k];";
            }

            src.new_line() << "return d * y / r;";
            src.close("}");
        }

        inline static void def_bspline(vex::backend::source_generator &src) {
            src.function<double>("Bspline") <<
                "(int k, double t) {\n"
                "    switch (k) {\n"
                "        case 0:\n"
                "            return (t * (t * (-t + 3.0) - 3.0) + 1.0) / 6.0;\n"
                "        case 1:\n"
                "            return (t * t * (3.0 * t - 6.0) + 4.0) / 6.0;\n"
                "        case 2:\n"
                "            return (t * (t * (-3.0 * t + 3.0) + 3.0) + 1.0) / 6.0;\n"
                "        case 3:\n"
                "            return t * t * t / 6.0;\n"
                "        default:\n"
                "            return 0.0;\n"
                "    }\n"
                "}\n\n"
                ;
        }

    private:
        double radius, scale, x0, hinv;
        std::vector<double> phi;

        static inline double Bspline(int k, double t) {
            assert(0 <= t && t < 1);
            assert(k < 4);

            switch (k) {
                case 0:
                    return (t * (t * (-t + 3.0) - 3.0) + 1.0) / 6.0;
                case 1:
                    return (t * t * (3.0 * t - 6.0) + 4.0) / 6.0;
                case 2:
                    return (t * (t * (-3.0 * t + 3.0) + 3.0) + 1.0) / 6.0;
                case 3:
                    return t * t * t / 6.0;
                default:
                    return 0.0;
            }
        }

};

class lennard_jones : public potential {
    public:
        lennard_jones(
                std::vector<double> x,
                std::vector<double> y,
                double xmin   = 0,
                double xmax   = 5,
                int grid_size = 8
                )
            : potential(x, residual(x, y), xmin, xmax, grid_size)
        {}

        double approx(double x) const {
            return impl(x);
        }

        void def_approx(vex::backend::source_generator &src, std::string name) const {
            src.function<double>(name + "_approx") <<
                "(double x) {\n"
                "    if (x < 1e-3) x = 1e-3;\n"
                "    double r = 1 / x;\n"
                "    double r3 = r * r * r;\n"
                "    double r6 = r3 * r3;\n"
                "    return 12 * r6 * (r6 - 1) * r;\n"
                "}\n\n"
                ;
        }
    private:
        inline static double impl(double x) {
            double r = 1 / x;
            double r3 = r * r * r;
            double r6 = r3 * r3;
            return 12 * r6 * (r6 - 1) * r;
        }

        static std::vector<double> residual(const std::vector<double> &x, std::vector<double> y) {
            for(unsigned i = 0; i < x.size(); ++i) {
                y[i] -= impl(x[i]);
            }
            return y;
        }
};

class soft_sphere : public potential {
    public:
        soft_sphere(
                std::vector<double> x,
                std::vector<double> y,
                double xmin   = 0,
                double xmax   = 5,
                int grid_size = 8
                )
            : potential(x, residual(x, y), xmin, xmax, grid_size)
        {}

        double approx(double x) const {
            return impl(x);
        }

        void def_approx(vex::backend::source_generator &src, std::string name) const {
            src.function<double>(name + "_approx") <<
                "(double x) {\n"
                "    if (x < 1e-3) x = 1e-3;\n"
                "    double r = 1 / x;\n"
                "    double r3 = r * r * r;\n"
                "    double r6 = r3 * r3;\n"
                "    return 12 * r6 * r6 * r;\n"
                "}\n\n"
                ;
        }
    private:
        inline static double impl(double x) {
            double r = 1 / x;
            double r3 = r * r * r;
            double r6 = r3 * r3;
            return 12 * r6 * r6 * r;
        }

        static std::vector<double> residual(const std::vector<double> &x, std::vector<double> y) {
            for(int i = 0; i < x.size(); ++i) {
                y[i] -= impl(x[i]);
            }
            return y;
        }
};

#endif
