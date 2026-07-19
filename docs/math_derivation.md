# VisionTrack — Kinematic Derivations

## 1. From pixels to meters
Given two reference points a known real distance `d` apart:
    meters_per_pixel = d / sqrt((bx-ax)^2 + (by-ay)^2)

World position (origin at launch point, y flipped so "up" is positive):
    x_world = (x_px - origin_x_px) * meters_per_pixel
    y_world = (origin_y_px - y_px) * meters_per_pixel

## 2. Velocity (central difference)
    v_i = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
This is a second-order accurate numerical derivative — far less noisy
than a naive forward difference (x[i+1]-x[i])/dt. `numpy.gradient`
implements this, with one-sided differences at the endpoints.

    vx(t) = dx/dt        vy(t) = dy/dt
    v(t)  = sqrt(vx(t)^2 + vy(t)^2)     (Pythagorean combination)

## 3. Acceleration
    ax(t) = dvx/dt        ay(t) = dvy/dt
For an ideal projectile (no drag), ax ≈ 0 and ay ≈ -g ≈ -9.81 m/s²,
which is a useful sanity check against measurement noise.

## 4. Launch angle
    theta_0 = atan2(vy[0], vx[0])

## 5. Max height, range, time of flight (measured)
- Landing frame = first frame after the trajectory's peak where
  y(t) returns to ≤ the launch height.
- Max height  H = max(y) - y[0]
- Range       R = x[landing] - x[0]
- Time of flight T = t[landing] - t[0]

## 6. Theoretical (ideal, no-drag) comparison
Using the MEASURED launch speed v0 and angle theta:
    T_theory = 2 * v0 * sin(theta) / g
    H_theory = (v0 * sin(theta))^2 / (2g)
    R_theory = v0^2 * sin(2*theta) / g

Comparing measured vs. theoretical values reveals the effect of air
resistance, spin, and measurement error — this is pure classical
physics, not a fitted/learned model.