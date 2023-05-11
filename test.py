import numpy as np
np.set_printoptions(precision=5, linewidth=np.inf)

t_co = np.arange(0.012, 0.1, 0.01)

s_n = 100240 
bw = 1e6 
snr = np.exp((s_n/bw) * np.log(2)/t_co) - 1
print(f"snr = {snr}")
print(f"t_co = {t_co}")

print("*"*80)

powers = np.arange(0.01, 0.101, 0.01)
for power in powers: 
    gains_over_N0 = snr/power
    print(f"power = {power}\ngains_over_N0 = {gains_over_N0}")
    e_co = power*t_co
    print(f"e_co = {e_co}")

print("*"*80)

t_cp = 2/3 * t_co
t_total = t_co + t_cp
print(f"t_cp = {t_cp}")
print(f"t_total = {t_total}")

print("*"*80)

C_n = 1e4 
D_n = np.arange(50, 350, 50)
print(f"D_n = {D_n}")
freqs = np.arange(0.5, 2.1, 0.25) * 1e9 
print(f"freqs = {freqs}")

kappa = 1e-28 
print("*"*80)
for freq in freqs: 
    print(f"freq = {freq:.2e}")
    for dn in D_n: 
        i = t_cp * freq / (C_n * dn)
        print(f"dn = {dn}\ni = {i}")
        e_cp = i * kappa * C_n * dn * (freq**2)
        print(f"e_cp = {e_cp}")
    print("*"*60)

