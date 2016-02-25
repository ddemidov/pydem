#ifndef DEM_HPP
#define DEM_HPP

#include <vector>
#include <memory>
#include <type_traits>

#include <vexcl/vexcl.hpp>

#include <boost/range.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

#include "potential.hpp"

typedef cl_double2 vec2d;
typedef std::array<int,2> int2int;

class dem {
    public:
        dem(
                int npoints, int nconn,
                const std::vector< std::vector< std::shared_ptr<potential> > > &P,
                const vec2d *_coord, const double *_mass,
                const double *_radius, const int *_type,
                const int2int *_conn,
                double cell_size = 2.0,
                double stiffness = 1e6,
                double grav      = 9.8,
                double gamma     = 1e-1,
                std::string aux_force = "0",
                std::string device = ""
                )
            :
            npoints(npoints), nconn(nconn), cell_size(cell_size),
            stiffness(stiffness), grav(grav), gamma(gamma),
            ctx(vex::Filter::Exclusive(
                        vex::Filter::Env &&
                        vex::Filter::Name(device) &&
                        vex::Filter::DoublePrecision &&
                        vex::Filter::Count(1))),
            coord(ctx, npoints), vel(ctx, npoints), acc(ctx, npoints),
            mass(ctx, npoints, _mass), radius(ctx, npoints, _radius),
            type(ctx, npoints, _type),
            cell_coo(ctx, npoints), cell_idx(ctx, npoints),
            part_ord(ctx, npoints), min(ctx), max(ctx)
        {
            pmin = {{ min(X(coord)), min(Y(coord)) }};
            pmax = {{ max(X(coord)), max(Y(coord)) }};

            compile_force_kernel(P, aux_force);
            compile_freeze_kernel();

            if (nconn) init_conn(_coord, _conn);
        }

        std::string device() const {
            std::ostringstream s;
            s << ctx.queue(0);
            return s.str();
        }

        double advance(
                vec2d *_coord,
                vec2d *_vel,
                vec2d *_acc,
                int    nsteps = 100,
                double dt     = 5e-4,
                double time   = 0.0
                )
        {
            namespace odeint = boost::numeric::odeint;

            vex::copy(_coord, _coord + npoints, coord.begin());
            vex::copy(_vel,   _vel   + npoints, vel.begin());
            vex::copy(_acc,   _acc   + npoints, acc.begin());

            for(int i = 0; i < nsteps; ++i, time += dt) {
                stepper.do_step(std::ref(*this), coord, vel, acc, coord, vel, acc, time, dt);
                freeze(ctx.queue(0), npoints, pmin, pmax, coord(0), vel(0), type(0));
            }

            vex::copy(coord.begin(), coord.end(), _coord);
            vex::copy(vel.begin(),   vel.end(),   _vel);
            vex::copy(acc.begin(),   acc.end(),   _acc);

            return time;
        }

        void operator()(
                vex::vector<vec2d> const &p,
                vex::vector<vec2d> const &v,
                vex::vector<vec2d>       &a,
                double time
                )
        {
            using namespace vex;

            // Compute bounding domain consisting of cells.
            vec2d p_min, p_max;
            std::tie(p_min, p_max) = get_domain(p);

            int ncx = (p_max.s[0] - p_min.s[0]) / cell_size;
            int ncy = (p_max.s[1] - p_min.s[1]) / cell_size;
            int nc  = ncx * ncy;

            if (cell_ptr.size() != nc + 1) {
                cell_ptr.resize(ctx, nc + 1);
                cell_ptr[0] = 0;
            }

            // Assign each particle to a cell, reset ordering.
            cell_coo = convert_int2( (p - p_min) / cell_size );
            cell_idx = cell_hash(cell_coo, ncx);
            part_ord = element_index();

            // Sort particle numbers in part_ord by cell idx.
            sort_by_key(cell_idx, part_ord);

            // Find range of each cell in sorted cell_idx array.
            permutation(element_index(1, nc))(cell_ptr) = hi_bound(
                    raw_pointer(cell_idx), npoints, element_index(0, nc));

            // Compute acceleration
            compute_force(ctx.queue(0),
                    npoints, ncx, nc, time, cell_coo(0), part_ord(0),
                    cell_ptr(0), p(0), v(0), mass(0), radius(0),
                    type(0), a(0)
                    );

            // Apply Hooke's Law.
            if (nconn) {
                permutation(body.idx)(a) += hookes_law(
                        element_index(), body.idx, stiffness,
                        raw_pointer(type),
                        raw_pointer(body.ptr),
                        raw_pointer(body.conn),
                        raw_pointer(body.dist),
                        raw_pointer(p)
                        ) / mass;
            }
        }
    private:
        typedef cl_int  hash_type;
        typedef cl_int2 index_type;

        int npoints, nconn;
        double cell_size, stiffness, grav, gamma;

        vex::Context ctx;
        vex::vector<vec2d>   coord;
        vex::vector<vec2d>   vel;
        vex::vector<vec2d>   acc;
        vex::vector<double>  mass;
        vex::vector<double>  radius;
        vex::vector<int>     type;

        vex::vector<index_type> cell_coo;
        vex::vector<hash_type>  cell_idx;
        vex::vector<hash_type>  part_ord;
        vex::vector<hash_type>  cell_ptr;

        vex::Reductor<double, vex::MIN> min;
        vex::Reductor<double, vex::MAX> max;

        cl_double2 pmin, pmax;

        struct {
            vex::vector<int>    idx;
            vex::vector<int>    ptr;
            vex::vector<int>    conn;
            vex::vector<double> dist;
        } body;

        vex::backend::kernel compute_force;
        vex::backend::kernel freeze;

        boost::numeric::odeint::velocity_verlet<
            vex::vector<vec2d>, vex::vector<vec2d>, double
            > stepper;

        VEX_FUNCTION(hash_type, cell_hash, (index_type, c)(cl_int, nx),
            return c.y * nx + c.x;
        );

        VEX_FUNCTION(hash_type, hi_bound, (const hash_type*, x)(size_t, n)(size_t, v),
            size_t begin = 0;
            size_t end   = n;
            while(end > begin) {
                size_t mid = begin + (end - begin) / 2;
                if (x[mid] <= v)
                    begin = ++mid;
                else
                    end = mid;
            }
            return begin;
        );

        VEX_FUNCTION(vec2d, hookes_law,
            (int,           i)
            (int,           idx)
            (double,        K)
            (const int*,    type)
            (const int*,    ptr)
            (const int*,    conn)
            (const double*, dist)
            (const vec2d*,  P),

            double2 F = {0.0, 0.0};

            if (type[idx] != 0) {
                double2 x = P[idx];
                for(int j = ptr[i], e = ptr[i + 1]; j < e; ++j) {
                    double2 y  = P[conn[j]];
                    double  d0 = dist[j];
                    double2 r  = y - x;
                    double  d  = length(r);

                    F += K * (r * (d - d0) / d);
                }
            }

            return F;
        );

        VEX_FUNCTION(double, X, (vec2d, p), return p.x; );
        VEX_FUNCTION(double, Y, (vec2d, p), return p.y; );

        std::tuple<vec2d, vec2d> get_domain(vex::vector<vec2d> const &p) const
        {
            vec2d p_min = {{ min(X(p)), min(Y(p)) }};
            vec2d p_max = {{ max(X(p)), max(Y(p)) }};

            double spanx = p_max.s[0] - p_min.s[0];
            double spany = p_max.s[1] - p_min.s[1];

            double dx = cell_size - fmod(spanx, cell_size);
            double dy = cell_size - fmod(spany, cell_size);

            p_min.s[0] -= dx / 2 + cell_size;
            p_min.s[1] -= dy / 2 + cell_size;
            p_max.s[0] += dx / 2 + cell_size;
            p_max.s[1] += dy / 2 + cell_size;

            return std::make_tuple(p_min, p_max);
        }

        static double distance(vec2d p1, vec2d p2) {
            double dx = p1.s[0] - p2.s[0];
            double dy = p1.s[1] - p2.s[1];
            return sqrt(dx * dx + dy * dy);
        }

        void init_conn(
                const vec2d *_coord,
                const int2int *_conn
                )
        {
            // Prepare body structure.
            const int* conn_data = reinterpret_cast<const int*>(_conn);
            std::vector<int> h_id(conn_data, conn_data + 2 * nconn);

            std::sort(h_id.begin(), h_id.end());
            h_id.erase(std::unique(h_id.begin(), h_id.end()), h_id.end());

            size_t nbody = h_id.size();

            std::vector<int> b_id(npoints);
            std::vector<int> h_ptr(nbody + 1, 0);

            for(unsigned i = 0; i < nbody; ++i)
                b_id[h_id[i]] = i;

            for(int i = 0; i < nconn; ++i) {
                ++h_ptr[1 + b_id[_conn[i][0]]];
                ++h_ptr[1 + b_id[_conn[i][1]]];
            }

            std::partial_sum(h_ptr.begin(), h_ptr.end(), h_ptr.begin());

            body.idx.resize(ctx, h_id);
            body.ptr.resize(ctx, h_ptr);

            std::vector<int>    h_conn(h_ptr.back());
            std::vector<double> h_dist(h_ptr.back());

            for(int k = 0; k < nconn; ++k) {
                int i  = _conn[k][0];
                int j  = _conn[k][1];
                int ii = b_id[i];
                int jj = b_id[j];

                double d = distance(_coord[i], _coord[j]);

                h_conn[h_ptr[ii]] = j;
                h_dist[h_ptr[ii]] = d;

                h_conn[h_ptr[jj]] = i;
                h_dist[h_ptr[jj]] = d;

                ++h_ptr[ii];
                ++h_ptr[jj];
            }

            body.conn.resize(ctx, h_conn);
            body.dist.resize(ctx, h_dist);
        }

        void compile_force_kernel(
                const std::vector< std::vector< std::shared_ptr<potential> > > &P,
                const std::string &aux_force
                )
        {
            using namespace vex;
            backend::source_generator src(ctx.queue(0));

            src.function<vec2d>("local_force").open("(")
                .parameter<vec2d >("pos")
                .parameter<vec2d >("vel")
                .parameter<int   >("type")
                .parameter<double>("mass")
                .parameter<double>("time")
                .close(")").open("{");

            src.new_line() << "if (type == 0) return (double2)(0.0, 0.0);";
            src.new_line() << "return -(" << gamma << ") * vel - mass * (double2)(0.0, " << grav << ") + " << aux_force << ";";
            src.close("}\n");

            potential::def_bspline(src);

            int nt = P.size();
            for (int i = 0; i < nt; ++i) {
                precondition(P[i].size() == nt, "potentials should be a square matrix");
                for(int j = 0; j < nt; ++j) {
                    P[i][j]->def_potential(src, "potential_" + std::to_string(i) + "_" + std::to_string(j));
                }
            }

            src.function<vec2d>("interaction_force").open("(")
                .parameter<vec2d>("d")
                .parameter<int  >("type1")
                .parameter<int  >("type2")
                .close(")").open("{");

            src.new_line() << "double2 f = {0.0, 0.0};";
            src.new_line() << "switch (type1)";
            src.open("{");
            src.new_line() << "case " << 0 << ": return (double2)(0.0, 0.0);";

            for(int i = 1; i < nt; ++i) {
                src.new_line() << "case " << i << ":";
                src.new_line() << "switch (type2)";
                src.open("{");
                for(int j = 0; j < nt; ++j) {
                    src.new_line() << "case " << j << ":";
                    src.new_line() << "    f = potential_" << i << "_" << j << "(d); break;";
                }
                src.close("}");
                src.new_line() << "break;";
            }
            src.close("}");
            src.new_line() << "return f;";
            src.close("}");

            src.kernel("compute_force").open("(")
                .parameter<int>("n")
                .parameter<int>("ncx")
                .parameter<int>("nc")
                .parameter<double>("time")
                .parameter<global_ptr<const cl_int2>>("cell_coo")
                .parameter<global_ptr<const int>>("part_ord")
                .parameter<global_ptr<const int>>("cell_ptr")
                .parameter<global_ptr<const cl_double2>>("P")
                .parameter<global_ptr<const cl_double2>>("V")
                .parameter<global_ptr<const double>>("M")
                .parameter<global_ptr<const double>>("R")
                .parameter<global_ptr<const int>>("T")
                .parameter<global_ptr<cl_double2>>("acc")
                .close(")").open("{");

            src.grid_stride_loop().open("{");
            src.new_line() << "int t = T[idx];";
            src.new_line() << "double2 p = P[idx];";
            src.new_line() << "double m = M[idx];";
            src.new_line() << "double r = R[idx];";
            src.new_line() << "int2 index = cell_coo[idx];";
            src.new_line() << "double2 f = local_force(p, V[idx], t, m, time);";
            src.new_line() << "for(int i = -1; i <= 1; ++i)"; src.open("{");
            src.new_line() << "for(int j = -1; j <= 1; ++j)"; src.open("{");
            src.new_line() << "int2 cell_index = index + (int2)(i,j);";
            src.new_line() << "int cell_hash = cell_index.y * ncx + cell_index.x;";
            src.new_line() << "if (cell_hash >= nc) continue;";
            src.new_line() << "for(int ii = cell_ptr[cell_hash], ee = cell_ptr[cell_hash+1]; ii < ee; ++ii)"; src.open("{");
            src.new_line() << "int jj = part_ord[ii];";
            src.new_line() << "if (jj == idx) continue;";
            src.new_line() << "f += interaction_force((p - P[jj]) / (r + R[jj]), t, T[jj]);";
            src.close("}").close("}").close("}");
            src.new_line() << "acc[idx] = f / m;";
            src.close("}").close("}");

            compute_force = backend::kernel(ctx.queue(0), src.str(), "compute_force");
        }

        void compile_freeze_kernel() {
            using namespace vex;
            backend::source_generator src(ctx.queue(0));

            src.kernel("freeze").open("(")
                .parameter<int>("n")
                .parameter<cl_double2>("pmin")
                .parameter<cl_double2>("pmax")
                .parameter<global_ptr<cl_double2>>("P")
                .parameter<global_ptr<cl_double2>>("V")
                .parameter<global_ptr<int>>("T")
                .close(")").open("{")
                .grid_stride_loop("idx").open("{");

            src.new_line() << "int t = T[idx];";
            src.new_line() << "if (t == 0) continue;";
            src.new_line() << "double2 p = P[idx];";

            src.new_line() << "if (p.x < pmin.x) t = 0;";
            src.new_line() << "if (p.x > pmax.x) t = 0;";
            src.new_line() << "if (p.y < pmin.y) t = 0;";
            src.new_line() << "if (p.y > pmax.y) t = 0;";

            src.new_line() << "if (t == 0) { T[idx] = 0; P[idx] = 0; V[idx] = 0; }";
            src.close("}").close("}");

            freeze = backend::kernel(ctx.queue(0), src.str(), "freeze");
        }
};

#endif
