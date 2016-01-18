#include <iostream>
#include <sstream>

#include <boost/range/iterator_range.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "potential.hpp"
#include "dem.hpp"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(potential, std::shared_ptr<potential>);

struct py_dem {
    py_dem(
            const std::vector< std::vector< std::shared_ptr<potential> > > &P,
            py::array_t<double> coord_buf,
            py::array_t<double> mass_buf,
            py::array_t<double> radius_buf,
            py::array_t<double> scale_buf,
            py::array_t<int>    type_buf,
            py::array_t<int>    conn_buf,
            double cell_size = 2.0,
            double stiffness = 1e6,
            double grav      = 9.8,
            double gamma     = 1e-1,
            std::string aux_force = std::string("0"),
            std::string device = std::string()
          )
    {
        py::buffer_info coord_info  = coord_buf.request();
        py::buffer_info mass_info   = mass_buf.request();
        py::buffer_info radius_info = radius_buf.request();
        py::buffer_info scale_info  = scale_buf.request();
        py::buffer_info type_info   = type_buf.request();
        py::buffer_info conn_info   = conn_buf.request();

        precondition(coord_info.ndim  == 2, "coord should be a 2D array");
        precondition(mass_info.ndim   == 1, "mass should be a 1D array");
        precondition(radius_info.ndim == 1, "radius should be a 1D array");
        precondition(scale_info.ndim  == 1, "scale should be a 1D array");
        precondition(type_info.ndim   == 1, "type should be a 1D array");
        precondition(conn_info.ndim   == 2, "conn should be a 2D array");

        precondition(coord_info.shape[1] == 2, "coord should be an (n x 2) array");
        precondition(conn_info.shape[1] == 2,  "conn should be an (n x 2) array");

        int np = coord_info.shape[0];

        precondition(mass_info.shape[0]   == np, "mass.shape[0] != num_points");
        precondition(radius_info.shape[0] == np, "radius.shape[0] != num_points");
        precondition(scale_info.shape[0]  == np, "scale.shape[0] != num_points");
        precondition(type_info.shape[0]   == np, "type.shape[0] != num_points");

        int nc = conn_info.shape[0];

        vec2d   *coord  = static_cast<vec2d*>(coord_info.ptr);
        double  *mass   = static_cast<double*>(mass_info.ptr);
        double  *radius = static_cast<double*>(radius_info.ptr);
        double  *scale  = static_cast<double*>(scale_info.ptr);
        int     *type   = static_cast<int*>(type_info.ptr);
        int2int *conn   = static_cast<int2int*>(conn_info.ptr);

        base = std::make_shared<dem>(np, nc, P, coord, mass, radius, scale,
                type, conn, cell_size, stiffness, grav, gamma, aux_force, device);
    }

    std::string device() const {
        return base->device();
    }

    double advance(
            py::array_t<double> coord_buf,
            py::array_t<double> vel_buf,
            py::array_t<double> acc_buf,
            int    nsteps = 100,
            double tau    = 5e-4,
            double t0     = 1.0
            )
    {
        py::buffer_info coord_info = coord_buf.request();
        py::buffer_info vel_info   = vel_buf.request();
        py::buffer_info acc_info   = acc_buf.request();

        precondition(coord_info.ndim == 2, "coord should be a 2D array");
        precondition(vel_info.ndim == 2,   "velocity should be a 2D array");
        precondition(acc_info.ndim == 2,   "acceleration should be a 2D array");

        precondition(coord_info.shape[1] == 2, "coord should be an (n x 2) array");
        precondition(vel_info.shape[1] == 2,   "vel should be an (n x 2) array");
        precondition(acc_info.shape[1] == 2,   "acc should be an (n x 2) array");

        int np = coord_info.shape[0];

        precondition(vel_info.shape[0] == np,  "velocity.shape[0] != num_points");
        precondition(acc_info.shape[0] == np,  "acceleration.shape[0] != num_points");

        vec2d   *coord = static_cast<vec2d*>(coord_info.ptr);
        vec2d   *vel   = static_cast<vec2d*>(vel_info.ptr);
        vec2d   *acc   = static_cast<vec2d*>(acc_info.ptr);

        return base->advance(coord, vel, acc, nsteps, tau, t0);
    }

    std::shared_ptr<dem> base;
};

PYBIND11_PLUGIN(pydem_ext) {
    py::module m("pydem_ext", "Particle method");

    py::class_<potential, std::shared_ptr<potential>>(m, "potential");

    py::class_<lennard_jones, std::shared_ptr<lennard_jones> >(m, "lennard_jones", py::base<potential>())
        .def(
                py::init<
                    std::vector<double>,
                    std::vector<double>,
                    double,
                    double,
                    int
                >(),
                "constructor",
                py::arg("x"),
                py::arg("y"),
                py::arg("xmin") = 0,
                py::arg("xmax") = 5,
                py::arg("grid_size") = 8
            )
        .def("__call__", [](const lennard_jones *p, py::array_t<double> x) {
                    auto apply = [p](double x) {
                        return (*p)(x);
                    };
                    return py::vectorize(apply)(x);
                }
            )
        .def("__str__", [](const lennard_jones *lj) { return "lennard_jones"; })
        ;

    py::class_<soft_sphere, std::shared_ptr<soft_sphere> >(m, "soft_sphere", py::base<potential>())
        .def(
                py::init<
                    std::vector<double>,
                    std::vector<double>,
                    double,
                    double,
                    int
                >(),
                "constructor",
                py::arg("x"),
                py::arg("y"),
                py::arg("xmin") = 0,
                py::arg("xmax") = 5,
                py::arg("grid_size") = 8
            )
        .def("__call__", [](const soft_sphere *p, py::array_t<double> x) {
                    auto apply = [p](double x) {
                        return (*p)(x);
                    };
                    return py::vectorize(apply)(x);
                }
            )
        .def("__str__", [](const soft_sphere *lj) { return "soft_sphere"; })
        ;

    py::class_<py_dem>(m, "dem")
        .def(
                py::init<
                    const std::vector< std::vector< std::shared_ptr<potential> > >&,
                    py::array_t<double>,
                    py::array_t<double>,
                    py::array_t<double>,
                    py::array_t<double>,
                    py::array_t<int>,
                    py::array_t<int>,
                    double,
                    double,
                    double,
                    double,
                    std::string,
                    std::string
                    >(),
                "constructor",
                py::arg("potential"),
                py::arg("coord"),
                py::arg("mass"),
                py::arg("radius"),
                py::arg("scale"),
                py::arg("types"),
                py::arg("conn"),
                py::arg("cell_size") = 2.0,
                py::arg("stiffness") = 1e6,
                py::arg("grav")      = 9.8,
                py::arg("gamma")     = 1e-1,
                py::arg("aux_force") = std::string("0"),
                py::arg("device") = std::string()
            )
        .def("device", &py_dem::device)
        .def("advance", &py_dem::advance,
                py::arg("coord"),
                py::arg("velocity"),
                py::arg("acceleration"),
                py::arg("nsteps") = 100,
                py::arg("tau")    = 5e-4,
                py::arg("t0")     = 0.0
            )
        ;

    return m.ptr();
}
