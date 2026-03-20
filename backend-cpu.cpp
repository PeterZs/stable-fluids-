#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

int clampi(const int value, const int lo, const int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

float clampf(const float value, const float lo, const float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
    return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
}

float fetch_clamped(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
    return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
}

float sample_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz) {
    gx = clampf(gx, 0.0f, static_cast<float>(sx - 1));
    gy = clampf(gy, 0.0f, static_cast<float>(sy - 1));
    gz = clampf(gz, 0.0f, static_cast<float>(sz - 1));
    const int x0 = clampi(static_cast<int>(std::floor(gx)), 0, sx - 1);
    const int y0 = clampi(static_cast<int>(std::floor(gy)), 0, sy - 1);
    const int z0 = clampi(static_cast<int>(std::floor(gz)), 0, sz - 1);
    const int x1 = std::min(x0 + 1, sx - 1);
    const int y1 = std::min(y0 + 1, sy - 1);
    const int z1 = std::min(z0 + 1, sz - 1);
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);
    const float c000 = field[index_3d(x0, y0, z0, sx, sy)];
    const float c100 = field[index_3d(x1, y0, z0, sx, sy)];
    const float c010 = field[index_3d(x0, y1, z0, sx, sy)];
    const float c110 = field[index_3d(x1, y1, z0, sx, sy)];
    const float c001 = field[index_3d(x0, y0, z1, sx, sy)];
    const float c101 = field[index_3d(x1, y0, z1, sx, sy)];
    const float c011 = field[index_3d(x0, y1, z1, sx, sy)];
    const float c111 = field[index_3d(x1, y1, z1, sx, sy)];
    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0 = c00 + (c10 - c00) * ty;
    const float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

float sample_scalar(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h - 0.5f, nx, ny, nz);
}

float sample_u(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
    return sample_grid(field, x / h, y / h - 0.5f, z / h - 0.5f, nx + 1, ny, nz);
}

float sample_v(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
    return sample_grid(field, x / h - 0.5f, y / h, z / h - 0.5f, nx, ny + 1, nz);
}

float sample_w(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h, nx, ny, nz + 1);
}

void clamp_domain(float& x, float& y, float& z, const int nx, const int ny, const int nz, const float h) {
    x = clampf(x, 0.0f, static_cast<float>(nx) * h);
    y = clampf(y, 0.0f, static_cast<float>(ny) * h);
    z = clampf(z, 0.0f, static_cast<float>(nz) * h);
}

void sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, float x, float y, float z, const int nx, const int ny, const int nz, const float h, float& out_x, float& out_y, float& out_z) {
    clamp_domain(x, y, z, nx, ny, nz, h);
    out_x = sample_u(velocity_x, x, y, z, nx, ny, nz, h);
    out_y = sample_v(velocity_y, x, y, z, nx, ny, nz, h);
    out_z = sample_w(velocity_z, x, y, z, nx, ny, nz, h);
}

void set_u_boundary(float* velocity_x, const int nx, const int ny, const int nz) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y) {
            velocity_x[index_3d(0, y, z, nx + 1, ny)] = 0.0f;
            velocity_x[index_3d(nx, y, z, nx + 1, ny)] = 0.0f;
        }
}

void set_v_boundary(float* velocity_y, const int nx, const int ny, const int nz) {
    for (int z = 0; z < nz; ++z)
        for (int x = 0; x < nx; ++x) {
            velocity_y[index_3d(x, 0, z, nx, ny + 1)] = 0.0f;
            velocity_y[index_3d(x, ny, z, nx, ny + 1)] = 0.0f;
        }
}

void set_w_boundary(float* velocity_z, const int nx, const int ny, const int nz) {
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            velocity_z[index_3d(x, y, 0, nx, ny)] = 0.0f;
            velocity_z[index_3d(x, y, nz, nx, ny)] = 0.0f;
        }
}

void advect_u(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x <= nx; ++x) {
                float px = static_cast<float>(x) * h;
                float py = (static_cast<float>(y) + 0.5f) * h;
                float pz = (static_cast<float>(z) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                destination[index_3d(x, y, z, nx + 1, ny)] = sample_u(source_x, px, py, pz, nx, ny, nz, h);
            }
}

void advect_v(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y <= ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * h;
                float py = static_cast<float>(y) * h;
                float pz = (static_cast<float>(z) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                destination[index_3d(x, y, z, nx, ny + 1)] = sample_v(source_y, px, py, pz, nx, ny, nz, h);
            }
}

void advect_w(float* destination, const float* source_x, const float* source_y, const float* source_z, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int z = 0; z <= nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * h;
                float py = (static_cast<float>(y) + 0.5f) * h;
                float pz = static_cast<float>(z) * h;
                float vx, vy, vz;
                sample_velocity(source_x, source_y, source_z, px, py, pz, nx, ny, nz, h, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                destination[index_3d(x, y, z, nx, ny)] = sample_w(source_z, px, py, pz, nx, ny, nz, h);
            }
}

void advect_scalar(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * h;
                float py = (static_cast<float>(y) + 0.5f) * h;
                float pz = (static_cast<float>(z) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, nx, ny, nz, h, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                destination[index_3d(x, y, z, nx, ny)] = std::max(0.0f, sample_scalar(source, px, py, pz, nx, ny, nz, h));
            }
}

void diffuse_grid(float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom, const int diffuse_iterations) {
    for (int iteration = 0; iteration < diffuse_iterations; ++iteration)
        for (int parity = 0; parity < 2; ++parity)
            for (int z = 0; z < sz; ++z)
                for (int y = 0; y < sy; ++y)
                    for (int x = 0; x < sx; ++x) {
                        if (((x + y + z) & 1) != parity) continue;
                        const float neighbors = fetch_clamped(destination, x - 1, y, z, sx, sy, sz) + fetch_clamped(destination, x + 1, y, z, sx, sy, sz) + fetch_clamped(destination, x, y - 1, z, sx, sy, sz) + fetch_clamped(destination, x, y + 1, z, sx, sy, sz)
                            + fetch_clamped(destination, x, y, z - 1, sx, sy, sz) + fetch_clamped(destination, x, y, z + 1, sx, sy, sz);
                        destination[index_3d(x, y, z, sx, sy)] = (source[index_3d(x, y, z, sx, sy)] + alpha * neighbors) / denom;
                    }
}

void compute_divergence(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz, const float h) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                divergence[index_3d(x, y, z, nx, ny)] = (fetch_clamped(velocity_x, x + 1, y, z, nx + 1, ny, nz) - fetch_clamped(velocity_x, x, y, z, nx + 1, ny, nz)
                    + fetch_clamped(velocity_y, x, y + 1, z, nx, ny + 1, nz) - fetch_clamped(velocity_y, x, y, z, nx, ny + 1, nz) + fetch_clamped(velocity_z, x, y, z + 1, nx, ny, nz + 1) - fetch_clamped(velocity_z, x, y, z, nx, ny, nz + 1)) / h;
}

void pressure_rbgs(float* pressure, const float* divergence, const int nx, const int ny, const int nz, const float h, const int parity) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                if (((x + y + z) & 1) != parity) continue;
                float sum = 0.0f;
                int count = 0;
                if (x > 0) {
                    sum += pressure[index_3d(x - 1, y, z, nx, ny)];
                    ++count;
                }
                if (x + 1 < nx) {
                    sum += pressure[index_3d(x + 1, y, z, nx, ny)];
                    ++count;
                }
                if (y > 0) {
                    sum += pressure[index_3d(x, y - 1, z, nx, ny)];
                    ++count;
                }
                if (y + 1 < ny) {
                    sum += pressure[index_3d(x, y + 1, z, nx, ny)];
                    ++count;
                }
                if (z > 0) {
                    sum += pressure[index_3d(x, y, z - 1, nx, ny)];
                    ++count;
                }
                if (z + 1 < nz) {
                    sum += pressure[index_3d(x, y, z + 1, nx, ny)];
                    ++count;
                }
                pressure[index_3d(x, y, z, nx, ny)] = count > 0 ? (sum - divergence[index_3d(x, y, z, nx, ny)] * h * h) / static_cast<float>(count) : 0.0f;
            }
}

void subtract_gradient_u(float* velocity_x, const float* pressure, const int nx, const int ny, const int nz, const float h) {
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 1; x < nx; ++x)
                velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x - 1, y, z, nx, ny)]) / h;
}

void subtract_gradient_v(float* velocity_y, const float* pressure, const int nx, const int ny, const int nz, const float h) {
    for (int z = 0; z < nz; ++z)
        for (int y = 1; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y - 1, z, nx, ny)]) / h;
}

void subtract_gradient_w(float* velocity_z, const float* pressure, const int nx, const int ny, const int nz, const float h) {
    for (int z = 1; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                velocity_z[index_3d(x, y, z, nx, ny)] -= (pressure[index_3d(x, y, z, nx, ny)] - pressure[index_3d(x, y, z - 1, nx, ny)]) / h;
}

} // namespace

extern "C" {

int32_t stable_fluids_step_cpu(const StableFluidsStepDesc* desc) {
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float viscosity = desc->viscosity;
    const float diffusion = desc->diffusion;
    const int32_t diffuse_iterations = desc->diffuse_iterations;
    const int32_t pressure_iterations = desc->pressure_iterations;

    auto* density_field = static_cast<float*>(desc->density);
    auto* density_temporary = static_cast<float*>(desc->temporary_density);
    auto* density_previous = static_cast<float*>(desc->temporary_previous_density);
    auto* velocity_x_field = static_cast<float*>(desc->velocity_x);
    auto* velocity_y_field = static_cast<float*>(desc->velocity_y);
    auto* velocity_z_field = static_cast<float*>(desc->velocity_z);
    auto* velocity_x_temporary = static_cast<float*>(desc->temporary_velocity_x);
    auto* velocity_y_temporary = static_cast<float*>(desc->temporary_velocity_y);
    auto* velocity_z_temporary = static_cast<float*>(desc->temporary_velocity_z);
    auto* velocity_x_previous = static_cast<float*>(desc->temporary_previous_velocity_x);
    auto* velocity_y_previous = static_cast<float*>(desc->temporary_previous_velocity_y);
    auto* velocity_z_previous = static_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure = static_cast<float*>(desc->temporary_pressure);
    auto* divergence = static_cast<float*>(desc->temporary_divergence);

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_x_field_bytes = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_y_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_z_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);

    std::memcpy(velocity_x_previous, velocity_x_field, velocity_x_field_bytes);
    std::memcpy(velocity_y_previous, velocity_y_field, velocity_y_field_bytes);
    std::memcpy(velocity_z_previous, velocity_z_field, velocity_z_field_bytes);
    advect_u(velocity_x_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
    advect_v(velocity_y_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
    advect_w(velocity_z_temporary, velocity_x_previous, velocity_y_previous, velocity_z_previous, nx, ny, nz, cell_size, dt);
    set_u_boundary(velocity_x_temporary, nx, ny, nz);
    set_v_boundary(velocity_y_temporary, nx, ny, nz);
    set_w_boundary(velocity_z_temporary, nx, ny, nz);

    if (viscosity <= 0.0f) {
        std::memcpy(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes);
        std::memcpy(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes);
        std::memcpy(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes);
    } else {
        std::memcpy(velocity_x_field, velocity_x_temporary, velocity_x_field_bytes);
        std::memcpy(velocity_y_field, velocity_y_temporary, velocity_y_field_bytes);
        std::memcpy(velocity_z_field, velocity_z_temporary, velocity_z_field_bytes);
        const float alpha = dt * viscosity / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        diffuse_grid(velocity_x_field, velocity_x_temporary, nx + 1, ny, nz, alpha, denom, diffuse_iterations);
        diffuse_grid(velocity_y_field, velocity_y_temporary, nx, ny + 1, nz, alpha, denom, diffuse_iterations);
        diffuse_grid(velocity_z_field, velocity_z_temporary, nx, ny, nz + 1, alpha, denom, diffuse_iterations);
        set_u_boundary(velocity_x_field, nx, ny, nz);
        set_v_boundary(velocity_y_field, nx, ny, nz);
        set_w_boundary(velocity_z_field, nx, ny, nz);
    }

    std::memset(pressure, 0, static_cast<std::size_t>(cell_bytes));
    compute_divergence(divergence, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size);
    for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
        pressure_rbgs(pressure, divergence, nx, ny, nz, cell_size, 0);
        pressure_rbgs(pressure, divergence, nx, ny, nz, cell_size, 1);
    }
    subtract_gradient_u(velocity_x_field, pressure, nx, ny, nz, cell_size);
    subtract_gradient_v(velocity_y_field, pressure, nx, ny, nz, cell_size);
    subtract_gradient_w(velocity_z_field, pressure, nx, ny, nz, cell_size);
    set_u_boundary(velocity_x_field, nx, ny, nz);
    set_v_boundary(velocity_y_field, nx, ny, nz);
    set_w_boundary(velocity_z_field, nx, ny, nz);

    std::memcpy(density_previous, density_field, static_cast<std::size_t>(cell_bytes));
    advect_scalar(density_temporary, density_previous, velocity_x_field, velocity_y_field, velocity_z_field, nx, ny, nz, cell_size, dt);

    if (diffusion <= 0.0f) {
        std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
    } else {
        std::memcpy(density_field, density_temporary, static_cast<std::size_t>(cell_bytes));
        const float alpha = dt * diffusion / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        diffuse_grid(density_field, density_temporary, nx, ny, nz, alpha, denom, diffuse_iterations);
    }

    return 0;
}

} // extern "C"
