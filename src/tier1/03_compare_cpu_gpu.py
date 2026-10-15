"""
Compara los outputs de 01_production_cpu_reference.py y 02_production_gpu.py.
Uso: correr ambos scripts primero (guardando logs), despues:
    python3 03_compare_cpu_gpu.py cpu_reference.log gpu_run.log
"""
import sys
import re

def parse_log(path):
    with open(path) as f:
        text = f.read()
    e_tot = float(re.search(r"E_tot = (-?\d+\.\d+)", text).group(1))
    tddft_energies = re.findall(r"Energias TDDFT \(eV\): \[(.*?)\]", text)[-1]
    energies = [float(x) for x in re.findall(r"[\d.]+", tddft_energies)]
    timings = dict(re.findall(r"\[timing\]\S* (.+?): ([\d.]+) s", text))
    return e_tot, energies, timings

if __name__ == "__main__":
    cpu_log, gpu_log = sys.argv[1], sys.argv[2]
    e_cpu, en_cpu, t_cpu = parse_log(cpu_log)
    e_gpu, en_gpu, t_gpu = parse_log(gpu_log)

    print("=== Control de correccion numerica ===")
    de = abs(e_cpu - e_gpu)
    print(f"E_tot CPU:  {e_cpu:.6f} Ha")
    print(f"E_tot GPU:  {e_gpu:.6f} Ha")
    print(f"Diferencia: {de:.2e} Ha  {'[OK, <1e-5]' if de < 1e-5 else '[REVISAR, >1e-5]'}")

    print("\nEnergias de excitacion TDDFT (eV):")
    for i, (ec, eg) in enumerate(zip(en_cpu, en_gpu)):
        diff = abs(ec - eg)
        flag = "OK" if diff < 0.01 else "REVISAR"
        print(f"  Estado {i+1}: CPU={ec:.3f}  GPU={eg:.3f}  diff={diff:.4f} eV  [{flag}]")

    print("\n=== Benchmark de performance ===")
    for key in t_cpu:
        if key in t_gpu:
            speedup = t_cpu[key] and (float(t_cpu[key]) / float(t_gpu[key]))
            print(f"  {key}: CPU={t_cpu[key]}s  GPU={t_gpu[key]}s  speedup={speedup:.1f}x")
