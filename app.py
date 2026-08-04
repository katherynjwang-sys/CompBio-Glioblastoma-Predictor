import os
import glob
import json
import time
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure, zoom
from skimage.measure import marching_cubes
import nibabel as nib

# =====================================================================
# PHASE 1: TIMEFRAME & CLINICAL DATA HANDLING
# =====================================================================

def resolve_predict_gbm_timeframe():
    """ Clinical planning delay between pre-op MRI and RT initiation (~14 days). """
    return 14.0

def load_nifti_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required volume file: {file_path}")
    nimg = nib.load(file_path)
    return nimg.get_fdata(), nimg.header.get_zooms()[:3], nimg.affine, nimg.header

def load_patient_pipeline_predict_gbm(patient_dir):
    print(f"--- Loading PREDICT-GBM Patient: {os.path.basename(patient_dir)} ---")
    files_to_load = {
        "t1gd": "t1c_bet_normalized.nii.gz",
        "flair": "flair_bet_normalized.nii.gz",
        "tumor_seg": "tumor_seg.nii.gz",
        "wm_map": "wm_pbmap.nii.gz",
        "gm_map": "gm_pbmap.nii.gz",
        "csf_map": "csf_pbmap.nii.gz"
    }
    
    patient_data = {}
    voxel_dims, affine, header = None, None, None
    for key, filename in files_to_load.items():
        full_path = os.path.join(patient_dir, filename)
        if os.path.exists(full_path):
            data, dims, aff, hdr = load_nifti_image(full_path)
            patient_data[key] = data
            if voxel_dims is None:
                voxel_dims, affine, header = dims, aff, hdr
        else:
            print(f"Warning: {filename} not found, substituting zeros.")
            
    # Handle missing CSF maps gracefully for legacy datasets
    if "csf_map" not in patient_data:
        patient_data["csf_map"] = np.zeros_like(patient_data.get("tumor_seg", np.zeros((10,10,10))))
            
    patient_data.update({
        "voxel_dims": voxel_dims, 
        "affine": affine, 
        "header": header, 
        "native_shape": patient_data["wm_map"].shape
    })
    
    seg = patient_data.get("tumor_seg", np.zeros_like(patient_data["wm_map"]))
    
    # Standard BraTS labels: 1=necrotic, 2=edema, 4=enhancing (sometimes 3)
    patient_data["t1c_core_mask"] = ((seg == 1) | (seg == 3) | (seg == 4))
    patient_data["flair_edema_mask"] = (seg > 0)
    
    patient_data["parenchyma_mask"] = ((patient_data["wm_map"] + patient_data["gm_map"]) > 0.1)
    patient_data["csf_skull_mask"] = (patient_data["csf_map"] > 0.1) | (~patient_data["parenchyma_mask"])
    
    # Synthesize Organs at Risk (OARs) if absent
    oar_path = os.path.join(patient_dir, "oar_mask.nii.gz")
    if os.path.exists(oar_path):
        patient_data["oar_mask"] = (load_nifti_image(oar_path)[0] > 0)
    else:
        oar_mask = np.zeros_like(seg, dtype=bool)
        coords = np.array(np.where(patient_data["wm_map"] > 0))
        if coords.size > 0:
            min_z = np.min(coords[2])
            # Calculate the anatomical center of mass in X and Y
            center_x = int(np.mean(coords[0]))
            center_y = int(np.mean(coords[1]))
            
            # Synthesize a 3D ellipsoid at the bottom-center to approximate the brainstem
            radius_x, radius_y, height_z = 15, 15, 25
            x_grid, y_grid, z_grid = np.ogrid[:seg.shape[0], :seg.shape[1], :seg.shape[2]]
            dist_from_center = ((x_grid - center_x)**2) / (radius_x**2) + \
                               ((y_grid - center_y)**2) / (radius_y**2) + \
                               ((z_grid - (min_z + height_z/2))**2) / (height_z**2)
            
            # Restrict the OAR entirely to the brain parenchyma boundary
            oar_mask = (dist_from_center <= 1) & patient_data["parenchyma_mask"]
        patient_data["oar_mask"] = oar_mask
    
    rec_path = os.path.join(patient_dir, "recurrence_preop.nii.gz")
    if os.path.exists(rec_path):
        rec_data = load_nifti_image(rec_path)[0]
        patient_data["resection_cavity"] = (rec_data == 4)
        patient_data["true_recurrence_mass"] = ((rec_data >= 1) & (rec_data <= 3))
    else:
        patient_data["resection_cavity"] = np.zeros_like(seg, dtype=bool)
        patient_data["true_recurrence_mass"] = np.zeros_like(seg, dtype=bool)
        
    return patient_data

# =====================================================================
# PHASE 2: LATTICE-BOLTZMANN SOLVER (D3Q7)
# =====================================================================

def perform_lbm_diffusion_step(u_current, f_current, neighbor_outside, D_map, dt, voxel_dims):
    """
    Executes a D3Q7 Lattice-Boltzmann collision/streaming step.
    Enforces exact bounce-back Neumann B.C. where glioblastoma hits CSF/Skull (D=0).
    """
    dirs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    inv = [0, 2, 1, 4, 3, 6, 5] 
    
    brain_mask = (D_map > 0.0)
    dx_mm = voxel_dims[0]
    
    D_lattice = D_map * (dt / (dx_mm ** 2))
    tau = np.where(brain_mask, 3.0 * D_lattice + 0.5, 0.501)
    tau = np.clip(tau, 0.505, 5.0)
    
    # Equilibrium (pseudo-D3Q7 for pure diffusion)
    f_eq = np.zeros_like(f_current)
    for i in range(1, 7):
        f_eq[i] = u_current / 6.0
        
    f_post = f_current - (f_current - f_eq) / tau[np.newaxis, ...]
    
    # Stream with strict bounce-back at CSF/Skull
    f_streamed = np.zeros_like(f_current)
    for i in range(1, 7):
        dx_i, dy_i, dz_i = dirs[i]
        streamed_incoming = np.roll(f_post[i], shift=(dx_i, dy_i, dz_i), axis=(0, 1, 2))
        
        # If the node from which we are trying to stream (opposite to direction i) is a wall, 
        # the particles that went towards that wall (f_post[inv[i]]) bounce back to us.
        f_streamed[i] = np.where(neighbor_outside[inv[i]], f_post[inv[i]], streamed_incoming)
        
    f_streamed[:, ~brain_mask] = 0.0
    u_diffused = np.sum(f_streamed[1:], axis=0)
    return np.clip(u_diffused, 0.0, 1.0), f_streamed

# =====================================================================
# PHASE 3: EXPONENTIAL MOLECULAR DOSIMETRY 
# =====================================================================

def optimize_prescription_dose(u_density, alpha_bar, d_int, voxel_dims):
    """
    Computes optimal dose ensuring total surviving cells are minimized for a given 
    integral dose constraint (d_int). Mathematically resolves to:
    d_i = max[0, (1/alpha_bar) * ln((u_i * alpha_bar) / mu)]
    """
    voxel_vol = np.prod(voxel_dims)
    mask = (u_density > 1e-5)
    
    def calculate_integrated_dose(mu_val):
        dose = np.zeros_like(u_density, dtype=np.float64)
        log_term = (u_density * alpha_bar) / mu_val
        valid = mask & (log_term > 1.0)
        if np.any(valid):
            dose[valid] = (1.0 / alpha_bar) * np.log(log_term[valid])
        np.clip(dose, 0.0, 60.0, out=dose)
        return np.sum(dose * voxel_vol)
        
    mu_low, mu_high = 1e-15, float(np.max(u_density) * alpha_bar)
    if mu_high <= mu_low: return np.zeros_like(u_density)
        
    for _ in range(60):
        mu_mid = (mu_low + mu_high) / 2.0
        if calculate_integrated_dose(mu_mid) > d_int:
            mu_low = mu_mid
        else:
            mu_high = mu_mid
            
    mu_opt = (mu_low + mu_high) / 2.0
    dose_grid = np.zeros_like(u_density, dtype=np.float64)
    log_term = (u_density * alpha_bar) / mu_opt
    valid = mask & (log_term > 1.0)
    if np.any(valid):
        dose_grid[valid] = (1.0 / alpha_bar) * np.log(log_term[valid])
        
    return np.clip(dose_grid, 0.0, 60.0)

# =====================================================================
# PHASE 4: SPATIAL UTILITIES & METROPOLIS-HASTINGS
# =====================================================================

def reconstruct_preop_density(coarse_maps, dw, rho, tau_1=0.80):
    t1c = np.asarray(coarse_maps['t1c_core_mask'], dtype=bool)
    voxel_dims = coarse_maps['voxel_dims']
    
    d_out = distance_transform_edt(~t1c, sampling=voxel_dims)
    d_in = distance_transform_edt(t1c, sampling=voxel_dims)
    
    lambda_val = max(0.1, float(np.sqrt(dw / max(rho, 1e-6))))
    u0 = np.zeros_like(t1c, dtype=np.float64)
    
    outside_t1c = (~t1c) & coarse_maps['parenchyma_mask']
    if np.any(outside_t1c):
        u0[outside_t1c] = tau_1 * np.exp(-d_out[outside_t1c] / lambda_val)
    if np.any(t1c):
        u0[t1c] = np.clip(tau_1 * np.exp(d_in[t1c] / lambda_val), 0.0, 1.0)
        
    return np.clip(u0, 0.0, 1.0)

def compute_95th_percentile_hausdorff(pred_mask, true_mask, voxel_dims):
    pred_mask, true_mask = np.asarray(pred_mask, dtype=bool), np.asarray(true_mask, dtype=bool)
    if not np.any(pred_mask) or not np.any(true_mask): return float('inf')
        
    struct = generate_binary_structure(3, 1)
    border_pred = pred_mask ^ binary_erosion(pred_mask, structure=struct)
    border_true = true_mask ^ binary_erosion(true_mask, structure=struct)
    
    dist_to_true = distance_transform_edt(~border_true, sampling=voxel_dims)
    dist_to_pred = distance_transform_edt(~border_pred, sampling=voxel_dims)
    
    dists1, dists2 = dist_to_true[border_pred], dist_to_pred[border_true]
    if len(dists1) == 0 and len(dists2) == 0: return float('inf')
    return float(np.percentile(np.concatenate([dists1, dists2]), 95))

def run_patient_simulation(dw, rho, patient_data, tau_1=0.80, target_time=14.0):
    u = reconstruct_preop_density(patient_data, dw, rho, tau_1=tau_1)
    dx_mm = patient_data['voxel_dims'][0]
    
    dt = min(0.02 * (dx_mm ** 2), target_time / 10.0)
    f = np.zeros((7,) + u.shape, dtype=np.float64)
    for i in range(1, 7): f[i] = u / 6.0
        
    # Tissue specific diffusion: Gray matter is 10% of White matter. CSF/Skull is blocked.
    D_map = patient_data['wm_map'] * dw + patient_data['gm_map'] * (dw / 10.0)
    csf_mask = np.asarray(patient_data.get('csf_skull_mask', np.zeros_like(u)), dtype=bool)
    cavity_mask = np.asarray(patient_data['resection_cavity'], dtype=bool)
    D_map[csf_mask | cavity_mask] = 0.0
    
    brain_mask = (D_map > 0.0)
    dirs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbor_outside = [~np.roll(brain_mask, shift=(-d[0], -d[1], -d[2]), axis=(0, 1, 2)) for d in dirs]
    
    steps = max(1, int(target_time / dt))
    for _ in range(steps):
        u_intermed, f = perform_lbm_diffusion_step(u, f, neighbor_outside, D_map, dt, patient_data['voxel_dims'])
        u = u_intermed + dt * (rho * u_intermed * (1.0 - u_intermed))
        u = np.clip(u, 0.0, 1.0)
        growth_scale = np.where(u_intermed > 1e-10, u / (u_intermed + 1e-12), 1.0)
        f = f * growth_scale[np.newaxis, ...]

        if not _ % 10:
            print(f"{_} / {steps}")
    return u

def calibrate_parameters_mh(patient_data, num_samples=300, tau_1=0.80, tau_2=0.16, sigma=5.0, target_time=14.0):
    print(f"-> Running Metropolis-Hastings (Scenario 1) for {num_samples} steps...")
    true_border_mask = np.asarray(patient_data['flair_edema_mask'], dtype=bool)
    
    def get_log_posterior(dw, rho):
        if not (0.01 <= dw <= 5.0 and 0.001 <= rho <= 0.5): return -np.inf
        u_0 = reconstruct_preop_density(patient_data, dw, rho, tau_1=tau_1)
        hd_95 = compute_95th_percentile_hausdorff(u_0 >= tau_2, true_border_mask, patient_data['voxel_dims'])
        return -1e6 if np.isinf(hd_95) else -(hd_95 ** 2) / (2 * (sigma ** 2))
            
    current_dw, current_rho = 0.1, 0.01
    current_lp = get_log_posterior(current_dw, current_rho)
    best_map = (current_dw, current_rho, current_lp)
    
    u_sum = np.zeros_like(true_border_mask, dtype=np.float64)
    u_sq_sum = np.zeros_like(true_border_mask, dtype=np.float64)
    accepted, burn_in = 0, int(num_samples * 0.3)
    
    for i in range(num_samples):
        prop_dw = np.exp(np.log(current_dw) + np.random.normal(0, 0.2))
        prop_rho = np.exp(np.log(current_rho) + np.random.normal(0, 0.2))
        prop_lp = get_log_posterior(prop_dw, prop_rho)
        
        if np.log(np.random.uniform()) < (prop_lp - current_lp):
            current_dw, current_rho, current_lp = prop_dw, prop_rho, prop_lp
            if current_lp > best_map[2]: best_map = (current_dw, current_rho, current_lp)
                
        print(f"   Iter {i+1}/{num_samples} | Dw: {current_dw:.4f}, Rho: {current_rho:.4f} | LP: {current_lp:.2f}")
        if i >= burn_in:
            u_t = run_patient_simulation(current_dw, current_rho, patient_data, tau_1, target_time)
            u_sum += u_t; u_sq_sum += (u_t ** 2); accepted += 1

    if accepted > 0:
        u_prob = u_sum / accepted
        u_std = np.sqrt(np.maximum((u_sq_sum / accepted) - (u_prob ** 2), 0.0))
    else:
        u_prob, u_std = np.zeros_like(true_border_mask, dtype=float), np.zeros_like(true_border_mask, dtype=float)
        
    u_map = run_patient_simulation(best_map[0], best_map[1], patient_data, tau_1, target_time)
    return u_prob, u_std, u_map

# =====================================================================
# PHASE 5: EXPORT ASSETS & CERR PLANNING
# =====================================================================

def export_wavefront_obj(filepath, volume, threshold, voxel_dims):
    if volume is None or not np.any(volume >= threshold): return False
    try:
        verts, faces, normals, _ = marching_cubes(volume, level=threshold, spacing=voxel_dims, step_size=1)
        with open(filepath, 'w') as f:
            for v in verts: f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for vn in normals: f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            for face in faces: f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        return True
    except Exception: return False

def export_dose_point_cloud(filepath, volume, threshold, voxel_dims):
    d0, d1, d2 = np.where(volume >= threshold)
    if len(d0) == 0: return False
    pts = []
    for idx in range(len(d0)):
        i, j, k = d0[idx], d1[idx], d2[idx]
        dose = float(volume[i, j, k])
        num_dots = int(np.clip(dose / 15.0, 1, 4))
        for _ in range(num_dots):
            jx, jy, jz = np.random.uniform(-0.5, 0.5, 3)
            pts.append([float((i+jx)*voxel_dims[0]), float((j+jy)*voxel_dims[1]), float((k+jz)*voxel_dims[2]), dose])
            
    if len(pts) > 150000: pts = [pts[idx] for idx in np.random.choice(len(pts), 150000, replace=False)]
    with open(filepath, 'w') as f: json.dump(pts, f)
    return True

def downsample_patient_data(patient_data, scale_factor=0.5):
    """
    Downsamples all spatial grids in patient_data to accelerate simulations.
    Uses nearest-neighbor (order=0) for masks and linear (order=1) for density maps.
    """
    downscaled = {}
    downscaled["voxel_dims"] = tuple(d / scale_factor for d in patient_data["voxel_dims"])
    downscaled["affine"] = patient_data["affine"]
    downscaled["header"] = patient_data["header"]
    downscaled["native_shape"] = patient_data["native_shape"]
    
    bool_keys = [
        "t1c_core_mask", "flair_edema_mask", "oar_mask", 
        "resection_cavity", "true_recurrence_mass", "parenchyma_mask", "csf_skull_mask"
    ]
    float_keys = ["wm_map", "gm_map", "csf_map"]
    
    for key in bool_keys:
        if key in patient_data:
            downscaled[key] = zoom(patient_data[key].astype(float), zoom=scale_factor, order=0) > 0.5
            
    for key in float_keys:
        if key in patient_data:
            downscaled[key] = zoom(patient_data[key], zoom=scale_factor, order=1)
            
    return downscaled


def upsample_array(array, target_shape, order=1):
    zoom_factors = [t / s for t, s in zip(target_shape, array.shape)]
    upscaled = zoom(array, zoom=zoom_factors, order=order)
    return np.clip(upscaled, 0.0, 1.0)

# =====================================================================
# PHASE 6: EXECUTION & WEB SERVER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Computational Oncology Pipeline")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Configuration mapped directly to Paper
    TAU_1, TAU_2, SIGMA = 0.80, 0.16, 5.0
    ALPHA_BAR = 0.05  # Corrected to exact paper constant (was 0.35)
    BETA_VAR = 1.0  # Weight for variance penalty in corrected dose
    GAMMA_OAR = 1.0 # Weight for OAR penalty (adjusted for standard penalization)
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "predict_gbm")
    patients = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
    processed_patients = []
    
    if not patients: print(f"Warning: No patients found at {base_dir}. Booting server empty.")

    for p_dir in patients:
        start_time = time.time()
        pid = os.path.basename(p_dir)
        try:
            maps = load_patient_pipeline_predict_gbm(p_dir)
            
            # --- 1. DOWNSCALE INPUT DATA ---
            scale_factor = 0.5  # Reduces memory and steps by up to 32x
            print(f"-> Downscaling grid by factor of {scale_factor} to accelerate simulation...")
            maps_down = downsample_patient_data(maps, scale_factor=scale_factor)
            
            # --- 2. RUN SIMULATION ON DOWNSCALED MAPS ---
            u_prob_down, u_std_down, u_map_down = calibrate_parameters_mh(
                maps_down, 15, TAU_1, TAU_2, SIGMA, resolve_predict_gbm_timeframe()
            )
            
            # --- 3. UPSCALE SIMULATION OUTPUTS BACK TO NATIVE SHAPE ---
            print("-> Rescaling spatial projections back to native high-resolution...")
            u_prob = upsample_array(u_prob_down, maps["native_shape"], order=1)
            u_std = upsample_array(u_std_down, maps["native_shape"], order=1)
            u_map = upsample_array(u_map_down, maps["native_shape"], order=1)
            
            # --- 4. PROCEED WITH DOSE CALCULATION ---
            print("-> Computing the 'Corrected' Dose Distribution...")
            oar_mask = maps.get('oar_mask', np.zeros_like(u_prob))
            u_corr = u_prob - (BETA_VAR * u_std) - (GAMMA_OAR * oar_mask)
            u_corr = np.clip(u_corr, 0.0, 1.0)
            
            for um in [u_map, u_prob, u_corr]:
                um[oar_mask] = 0.0
                um[~maps['parenchyma_mask']] = 0.0
            
            # Standard Plan limits 60 Gy to CTV (T1c + 20mm)
            t1c_mask = maps['t1c_core_mask'] > 0
            if np.any(t1c_mask):
                dist_t1c = distance_transform_edt(~t1c_mask, sampling=maps['voxel_dims'])
                ctv_mask = (dist_t1c <= 20.0) & maps['parenchyma_mask']
            else: ctv_mask = maps['flair_edema_mask'] & maps['parenchyma_mask']

            d_std = np.zeros_like(u_map)
            d_std[ctv_mask] = 60.0
            d_int_phys = np.sum(d_std * np.prod(maps['voxel_dims']))
            
            corr_dose = optimize_prescription_dose(u_corr, ALPHA_BAR, d_int_phys, maps['voxel_dims'])
            corr_dose[oar_mask] = 0.0
            corr_dose[~maps['parenchyma_mask']] = 0.0
            
            true_rec = np.asarray(maps['true_recurrence_mass'], dtype=bool)
            pred_rec = (u_prob >= TAU_2)
            hd = compute_95th_percentile_hausdorff(pred_rec, true_rec, maps['voxel_dims'])
            dice = (2.0 * np.sum(pred_rec & true_rec)) / (np.sum(pred_rec) + np.sum(true_rec) + 1e-8)
            
            # --- 5. COMPUTE HEALTHY CELL SURVIVAL SPARED METRICS ---
            v_vol = np.prod(maps['voxel_dims'])
            # Healthy cell tissue fraction = 1 - expected tumor cell density
            h_fraction = np.maximum(0.0, 1.0 - u_prob)
            
            # Evaluate only within brain parenchyma to prevent air volumes from skewing ratios
            brain_mask = maps['parenchyma_mask']
            surv_h_std = np.sum(h_fraction[brain_mask] * np.exp(-ALPHA_BAR * d_std[brain_mask]) * v_vol)
            surv_h_pers = np.sum(h_fraction[brain_mask] * np.exp(-ALPHA_BAR * corr_dose[brain_mask]) * v_vol)
            
            # Relative increase in surviving healthy brain cells
            h_increase_pct = ((surv_h_pers - surv_h_std) / max(surv_h_std, 1e-8)) * 100.0
            
            out = os.path.join("web_assets", pid)
            os.makedirs(out, exist_ok=True)
            
            metrics = {
                "Spatial_Accuracy": {"DSC": round(float(dice),4), "HD95": round(float(hd),2)},
                "Dosimetric_Superiority": {
                    "Standard_Healthy_Surv": round(float(surv_h_std),2), 
                    "Personalized_Healthy_Surv": round(float(surv_h_pers),2), 
                    "Healthy_Surv_Increase_Pct": round(float(h_increase_pct),2)
                }
            }
            with open(os.path.join(out, "metrics.json"), "w") as f: json.dump(metrics, f)
            
            v = maps['voxel_dims']
            export_wavefront_obj(os.path.join(out, "00_brain.obj"), maps['parenchyma_mask'], 0.5, v)
            export_wavefront_obj(os.path.join(out, "00_csf_skull.obj"), maps['csf_skull_mask'], 0.5, v)
            export_wavefront_obj(os.path.join(out, "00_oar.obj"), oar_mask, 0.5, v)
            export_wavefront_obj(os.path.join(out, "01_t1c.obj"), maps['t1c_core_mask'], 0.5, v)
            export_wavefront_obj(os.path.join(out, "02_flair.obj"), maps['flair_edema_mask'], 0.5, v)
            export_wavefront_obj(os.path.join(out, "03_predicted.obj"), u_map, TAU_2, v)
            export_dose_point_cloud(os.path.join(out, "04_dose.json"), corr_dose, 1.0, v)
            export_wavefront_obj(os.path.join(out, "06_recurrence.obj"), maps['true_recurrence_mass'], 0.5, v)
            
            processed_patients.append(pid)
            print(f"-> Saved: {pid} ({time.time()-start_time:.1f}s) - HD95: {hd:.2f}mm")
        except Exception as e:
            print(f"Failed {pid}: {e}")
            import traceback; traceback.print_exc()
            
    os.makedirs("web_assets", exist_ok=True)
    with open("web_assets/patients.json", "w") as f: json.dump(processed_patients, f)

    class CORS(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()
            
    print(f"\n[SERVER] Online at http://localhost:{args.port}/index.html")
    HTTPServer(('', args.port), CORS).serve_forever()

if __name__ == "__main__": main()