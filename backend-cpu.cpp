#include "stable-fluids.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {

int32_t stable_fluids_step_cpu(const StableFluidsStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(StableFluidsStepDesc)) return 1000;
    if (desc->stream != nullptr) return 3002;
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float viscosity = desc->viscosity;
    const float diffusion = desc->diffusion;
    const int32_t diffuse_iterations = desc->diffuse_iterations;
    const int32_t pressure_iterations = desc->pressure_iterations;
    if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
    if (cell_size <= 0.0f) return 1002;
    if (dt <= 0.0f) return 1003;
    if (diffuse_iterations <= 0 || pressure_iterations <= 0) return 1004;
    if (desc->density == nullptr) return 2001;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_density == nullptr) return 2007;
    if (desc->temporary_velocity_x == nullptr) return 2008;
    if (desc->temporary_velocity_y == nullptr) return 2009;
    if (desc->temporary_velocity_z == nullptr) return 2010;
    if (desc->temporary_previous_density == nullptr) return 2011;
    if (desc->temporary_previous_velocity_x == nullptr) return 2012;
    if (desc->temporary_previous_velocity_y == nullptr) return 2013;
    if (desc->temporary_previous_velocity_z == nullptr) return 2014;
    if (desc->temporary_pressure == nullptr) return 2015;
    if (desc->temporary_divergence == nullptr) return 2016;

    auto* density = reinterpret_cast<float*>(desc->density);
    auto* velocity_x = reinterpret_cast<float*>(desc->velocity_x);
    auto* velocity_y = reinterpret_cast<float*>(desc->velocity_y);
    auto* velocity_z = reinterpret_cast<float*>(desc->velocity_z);
    auto* temporary_density = reinterpret_cast<float*>(desc->temporary_density);
    auto* temporary_velocity_x = reinterpret_cast<float*>(desc->temporary_velocity_x);
    auto* temporary_velocity_y = reinterpret_cast<float*>(desc->temporary_velocity_y);
    auto* temporary_velocity_z = reinterpret_cast<float*>(desc->temporary_velocity_z);
    auto* temporary_previous_density = reinterpret_cast<float*>(desc->temporary_previous_density);
    auto* temporary_previous_velocity_x = reinterpret_cast<float*>(desc->temporary_previous_velocity_x);
    auto* temporary_previous_velocity_y = reinterpret_cast<float*>(desc->temporary_previous_velocity_y);
    auto* temporary_previous_velocity_z = reinterpret_cast<float*>(desc->temporary_previous_velocity_z);
    auto* temporary_pressure = reinterpret_cast<float*>(desc->temporary_pressure);
    auto* temporary_divergence = reinterpret_cast<float*>(desc->temporary_divergence);

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_x_field_bytes = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_y_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_z_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);
    auto clampi = [](const int value, const int lo, const int hi) { return value < lo ? lo : (value > hi ? hi : value); };
    auto clampf = [](const float value, const float lo, const float hi) { return value < lo ? lo : (value > hi ? hi : value); };
    auto index_3d = [](const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    };
    auto fetch_clamped = [&](const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
        return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
    };
    auto sample_grid = [&](const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz) {
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
    };
    auto sample_scalar = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size - 0.5f, z / cell_size - 0.5f, nx, ny, nz); };
    auto sample_u = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size, y / cell_size - 0.5f, z / cell_size - 0.5f, nx + 1, ny, nz); };
    auto sample_v = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size, z / cell_size - 0.5f, nx, ny + 1, nz); };
    auto sample_w = [&](const float* field, const float x, const float y, const float z) { return sample_grid(field, x / cell_size - 0.5f, y / cell_size - 0.5f, z / cell_size, nx, ny, nz + 1); };
    auto clamp_domain = [&](float& x, float& y, float& z) {
        x = clampf(x, 0.0f, static_cast<float>(nx) * cell_size);
        y = clampf(y, 0.0f, static_cast<float>(ny) * cell_size);
        z = clampf(z, 0.0f, static_cast<float>(nz) * cell_size);
    };
    auto sample_velocity = [&](const float* source_x, const float* source_y, const float* source_z, float x, float y, float z, float& out_x, float& out_y, float& out_z) {
        clamp_domain(x, y, z);
        out_x = sample_u(source_x, x, y, z);
        out_y = sample_v(source_y, x, y, z);
        out_z = sample_w(source_z, x, y, z);
    };
    auto set_boundaries = [&](float* u, float* v, float* w) {
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y) {
                u[index_3d(0, y, z, nx + 1, ny)] = 0.0f;
                u[index_3d(nx, y, z, nx + 1, ny)] = 0.0f;
            }
        for (int z = 0; z < nz; ++z)
            for (int x = 0; x < nx; ++x) {
                v[index_3d(x, 0, z, nx, ny + 1)] = 0.0f;
                v[index_3d(x, ny, z, nx, ny + 1)] = 0.0f;
            }
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                w[index_3d(x, y, 0, nx, ny)] = 0.0f;
                w[index_3d(x, y, nz, nx, ny)] = 0.0f;
            }
    };
    auto diffuse_grid = [&](float* destination, const float* source, const int sx, const int sy, const int sz, const float alpha, const float denom) {
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
    };

    std::memcpy(temporary_previous_velocity_x, velocity_x, velocity_x_field_bytes);
    std::memcpy(temporary_previous_velocity_y, velocity_y, velocity_y_field_bytes);
    std::memcpy(temporary_previous_velocity_z, velocity_z, velocity_z_field_bytes);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x <= nx; ++x) {
                float px = static_cast<float>(x) * cell_size;
                float py = (static_cast<float>(y) + 0.5f) * cell_size;
                float pz = (static_cast<float>(z) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(temporary_previous_velocity_x, temporary_previous_velocity_y, temporary_previous_velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                temporary_velocity_x[index_3d(x, y, z, nx + 1, ny)] = sample_u(temporary_previous_velocity_x, px, py, pz);
            }
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y <= ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * cell_size;
                float py = static_cast<float>(y) * cell_size;
                float pz = (static_cast<float>(z) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(temporary_previous_velocity_x, temporary_previous_velocity_y, temporary_previous_velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                temporary_velocity_y[index_3d(x, y, z, nx, ny + 1)] = sample_v(temporary_previous_velocity_y, px, py, pz);
            }
    for (int z = 0; z <= nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * cell_size;
                float py = (static_cast<float>(y) + 0.5f) * cell_size;
                float pz = static_cast<float>(z) * cell_size;
                float vx, vy, vz;
                sample_velocity(temporary_previous_velocity_x, temporary_previous_velocity_y, temporary_previous_velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                temporary_velocity_z[index_3d(x, y, z, nx, ny)] = sample_w(temporary_previous_velocity_z, px, py, pz);
            }
    set_boundaries(temporary_velocity_x, temporary_velocity_y, temporary_velocity_z);

    if (viscosity <= 0.0f) {
        std::memcpy(velocity_x, temporary_velocity_x, velocity_x_field_bytes);
        std::memcpy(velocity_y, temporary_velocity_y, velocity_y_field_bytes);
        std::memcpy(velocity_z, temporary_velocity_z, velocity_z_field_bytes);
    } else {
        std::memcpy(velocity_x, temporary_velocity_x, velocity_x_field_bytes);
        std::memcpy(velocity_y, temporary_velocity_y, velocity_y_field_bytes);
        std::memcpy(velocity_z, temporary_velocity_z, velocity_z_field_bytes);
        const float alpha = dt * viscosity / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        diffuse_grid(velocity_x, temporary_velocity_x, nx + 1, ny, nz, alpha, denom);
        diffuse_grid(velocity_y, temporary_velocity_y, nx, ny + 1, nz, alpha, denom);
        diffuse_grid(velocity_z, temporary_velocity_z, nx, ny, nz + 1, alpha, denom);
        set_boundaries(velocity_x, velocity_y, velocity_z);
    }

    std::memset(temporary_pressure, 0, static_cast<std::size_t>(cell_bytes));
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                temporary_divergence[index_3d(x, y, z, nx, ny)] = (fetch_clamped(velocity_x, x + 1, y, z, nx + 1, ny, nz) - fetch_clamped(velocity_x, x, y, z, nx + 1, ny, nz) + fetch_clamped(velocity_y, x, y + 1, z, nx, ny + 1, nz)
                    - fetch_clamped(velocity_y, x, y, z, nx, ny + 1, nz) + fetch_clamped(velocity_z, x, y, z + 1, nx, ny, nz + 1) - fetch_clamped(velocity_z, x, y, z, nx, ny, nz + 1)) / cell_size;
    for (int iteration = 0; iteration < pressure_iterations; ++iteration)
        for (int parity = 0; parity < 2; ++parity)
            for (int z = 0; z < nz; ++z)
                for (int y = 0; y < ny; ++y)
                    for (int x = 0; x < nx; ++x) {
                        if (((x + y + z) & 1) != parity) continue;
                        float sum = 0.0f;
                        int count = 0;
                        if (x > 0) {
                            sum += temporary_pressure[index_3d(x - 1, y, z, nx, ny)];
                            ++count;
                        }
                        if (x + 1 < nx) {
                            sum += temporary_pressure[index_3d(x + 1, y, z, nx, ny)];
                            ++count;
                        }
                        if (y > 0) {
                            sum += temporary_pressure[index_3d(x, y - 1, z, nx, ny)];
                            ++count;
                        }
                        if (y + 1 < ny) {
                            sum += temporary_pressure[index_3d(x, y + 1, z, nx, ny)];
                            ++count;
                        }
                        if (z > 0) {
                            sum += temporary_pressure[index_3d(x, y, z - 1, nx, ny)];
                            ++count;
                        }
                        if (z + 1 < nz) {
                            sum += temporary_pressure[index_3d(x, y, z + 1, nx, ny)];
                            ++count;
                        }
                        temporary_pressure[index_3d(x, y, z, nx, ny)] = count > 0 ? (sum - temporary_divergence[index_3d(x, y, z, nx, ny)] * cell_size * cell_size) / static_cast<float>(count) : 0.0f;
                    }
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 1; x < nx; ++x)
                velocity_x[index_3d(x, y, z, nx + 1, ny)] -= (temporary_pressure[index_3d(x, y, z, nx, ny)] - temporary_pressure[index_3d(x - 1, y, z, nx, ny)]) / cell_size;
    for (int z = 0; z < nz; ++z)
        for (int y = 1; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                velocity_y[index_3d(x, y, z, nx, ny + 1)] -= (temporary_pressure[index_3d(x, y, z, nx, ny)] - temporary_pressure[index_3d(x, y - 1, z, nx, ny)]) / cell_size;
    for (int z = 1; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                velocity_z[index_3d(x, y, z, nx, ny)] -= (temporary_pressure[index_3d(x, y, z, nx, ny)] - temporary_pressure[index_3d(x, y, z - 1, nx, ny)]) / cell_size;
    set_boundaries(velocity_x, velocity_y, velocity_z);

    std::memcpy(temporary_previous_density, density, cell_bytes);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float px = (static_cast<float>(x) + 0.5f) * cell_size;
                float py = (static_cast<float>(y) + 0.5f) * cell_size;
                float pz = (static_cast<float>(z) + 0.5f) * cell_size;
                float vx, vy, vz;
                sample_velocity(velocity_x, velocity_y, velocity_z, px, py, pz, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz);
                temporary_density[index_3d(x, y, z, nx, ny)] = std::max(0.0f, sample_scalar(temporary_previous_density, px, py, pz));
            }

    if (diffusion <= 0.0f) {
        std::memcpy(density, temporary_density, cell_bytes);
    } else {
        std::memcpy(density, temporary_density, cell_bytes);
        const float alpha = dt * diffusion / (cell_size * cell_size);
        const float denom = 1.0f + 6.0f * alpha;
        diffuse_grid(density, temporary_density, nx, ny, nz, alpha, denom);
    }

    return 0;
}

} // extern "C"
