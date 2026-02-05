import numpy as np
import matplotlib.pyplot as plt

# ==========================
# PARÂMETROS GERAIS
# ==========================
res = 200          # resolução menor pra testar
fov = 4.0          # "campo de visão"
cam_z = 8.0       # posição da câmera

M = 1.0
r_s = 2 * M        # horizonte
r_ph = 3 * M       # raio da órbita de fóton

r_disk_in = 3.2 * M
r_disk_out = 15 * M

max_steps = 1000   # bem menos que 4000
ds = 0.02          # passo um pouco maior

# ==========================
# GRADE DE PIXELS
# ==========================
xs = np.linspace(-fov, fov, res)
ys = np.linspace(-fov, fov, res)
XX, YY = np.meshgrid(xs, ys)

img = np.zeros((res, res, 3), dtype=np.float32)

# ==========================
# FUNÇÕES
# ==========================

M = 1.0
r_s = 2 * M

def grav_redshift(r):
    if r <= r_s:
        return 0.0
    return np.sqrt(1.0 - 2.0*M / r)

def disk_color_GR(r, phi, vel_phi=0.55):
    base = 1.0 / (r**3)
    base = np.clip(base, 0, 5)

    g = grav_redshift(r)
    if g <= 0:
        return np.array([0.0, 0.0, 0.0])

    v = vel_phi
    gamma = 1.0 / np.sqrt(1 - v**2)

    vx = -v * np.sin(phi)
    vy =  v * np.cos(phi)
    vz =  0.0
    v_vec = np.array([vx, vy, vz])

    px = r * np.cos(phi)
    py = r * np.sin(phi)
    pz = 0.0
    p = np.array([px, py, pz])
    cam = np.array([0.0, 0.0, cam_z])
    k_dir = cam - p
    k_dir = k_dir / np.linalg.norm(k_dir)

    vdotk = np.dot(v_vec, k_dir)
    D = 1.0 / (gamma * (1.0 - vdotk))
    D = np.clip(D, 0.2, 3.0)

    I = base * (g**2) * (D**3)
    I = np.clip(I, 0, 5.0)

    t = np.clip(I / 3.0, 0, 1)

    col_cool = np.array([0.5, 0.15, 0.02])
    col_hot  = np.array([1.0, 0.95, 0.85])

    col = (1 - t) * col_cool + t * col_hot
    return np.clip(col, 0, 1)

def background_color(direction):
    dz = direction[2]
    t = 0.5*(dz + 1.0)
    t = np.clip(t, 0, 1)

    col1 = np.array([0.01, 0.01, 0.04])
    col2 = np.array([0.15, 0.0, 0.2])

    return (1 - t)*col1 + t*col2

# ==========================
# LOOP DOS RAIOS
# ==========================

for i in range(res):
    # print de progresso pra você ver que não travou
    if i % 20 == 0:
        print(f"Linha {i+1}/{res}")

    for j in range(res):
        x = XX[i, j]
        y = YY[i, j]
        z = cam_z

        tilt = np.radians(35)  # inclinação da câmera

        # rotação simples em torno do eixo x
        # assim inclinamos o olhar "para baixo e um pouco pra frente"
        dz = -np.cos(tilt)
        dy =  np.sin(tilt)

        dir_vec = np.array([x, y*0.7 + dy, dz])
        dir_vec /= np.linalg.norm(dir_vec)

        pos = np.array([x, y, z], dtype=np.float64)
        vel = dir_vec.copy()

        hit_color = np.zeros(3, dtype=np.float32)

        for step in range(max_steps):
            pos += vel * ds
            r = np.linalg.norm(pos)

            if r <= r_s:
                hit_color = np.array([0.0, 0.0, 0.0])
                break

            if r > 60.0:
                hit_color = background_color(vel)
                break

            if r > 1e-3:
                to_center = -pos / r
                proj = np.dot(to_center, vel)
                F_perp = to_center - proj * vel

                k = 1.0 / (r**3)
                k *= 1.0 + 3.0*np.exp(-((r - r_ph)/0.7)**2)

                bend_strength = 1.4
                vel += bend_strength * k * F_perp * ds
                vel /= np.linalg.norm(vel)

            if abs(pos[2]) < 0.02:
                r_xy = np.linalg.norm(pos[:2])
                if r_disk_in < r_xy < r_disk_out:
                    phi = np.arctan2(pos[1], pos[0])
                    hit_color = disk_color_GR(r_xy, phi)
                    break

        img[i, j, :] = hit_color

# ==========================
# PLOT
# ==========================
img = np.clip(img, 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(img, origin="lower")
plt.axis("off")
plt.title("BH (versão light, Python)")
plt.tight_layout()
plt.show()
