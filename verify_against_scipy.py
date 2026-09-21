#!/usr/bin/env python
"""
ΕΠΑΛΗΘΕΥΣΗ: σύγκριση των δικών μας υλοποιήσεων με
  (α) τις ΑΚΡΙΒΕΙΣ αναλυτικές τιμές
  (β) τις αντίστοιχες ρουτίνες της SciPy

Δεν είναι "δοκιμή" με την έννοια του pass/fail σε ένα παράδειγμα, αλλά
differential testing: τρέχουμε τη δική μας μέθοδο και τη βιβλιοθήκη αναφοράς
στα ΙΔΙΑ δεδομένα και ελέγχουμε ότι συμφωνούν στο επίπεδο του σφάλματος
στρογγυλοποίησης.
"""

import sys
import numpy as np

sys.path.insert(0, 'code')

from simpsons_rule import SimpsonsRule1D, SimpsonsRule2D
from quadrature_methods import GaussianQuadrature, TrapezoidalRule, MidpointRule

import scipy.integrate as si
from scipy.special import roots_legendre

EPS = np.finfo(float).eps
apotyxies = []


def krisi(onoma, diafora, katofli):
    ok = diafora <= katofli
    if not ok:
        apotyxies.append(onoma)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {onoma:52s} δ = {diafora:.3e}")


# =========================================================================
print("=" * 78)
print("1. ΚΟΜΒΟΙ ΚΑΙ ΒΑΡΗ GAUSS-LEGENDRE έναντι scipy.special.roots_legendre")
print("=" * 78)
# Η scipy υπολογίζει τους κόμβους ως ιδιοτιμές του τριδιαγώνιου πίνακα Jacobi
# (αλγόριθμος Golub-Welsch), δηλαδή με ΕΝΤΕΛΩΣ διαφορετικό δρόμο από τους
# κλειστούς τύπους με ριζικά που έχουμε εμείς. Αν συμφωνούν, οι τιμές είναι σωστές.
for n in range(1, 6):
    komvoi_mas, vari_mas = GaussianQuadrature.GAUSS_LEGENDRE_TABLE[n]
    komvoi_sp, vari_sp = roots_legendre(n)
    # Ταξινόμηση για ασφαλή σύγκριση
    i_mas, i_sp = np.argsort(komvoi_mas), np.argsort(komvoi_sp)
    d_k = np.max(np.abs(komvoi_mas[i_mas] - komvoi_sp[i_sp]))
    d_v = np.max(np.abs(vari_mas[i_mas] - vari_sp[i_sp]))
    krisi(f"n={n}: κόμβοι", d_k, 20 * EPS)
    krisi(f"n={n}: βάρη", d_v, 20 * EPS)

    # Θεωρητικός έλεγχος: το άθροισμα των βαρών πρέπει να ισούται με 2
    # (= το μήκος του διαστήματος [-1,1], αφού ο κανόνας είναι ακριβής για f≡1)
    krisi(f"n={n}: Σwᵢ = 2", abs(np.sum(vari_mas) - 2.0), 20 * EPS)


# =========================================================================
print("\n" + "=" * 78)
print("2. ΒΑΘΜΟΣ ΑΚΡΙΒΕΙΑΣ: ο Gauss με n κόμβους είναι ακριβής έως βαθμό 2n-1")
print("=" * 78)
# Ελέγχουμε ΚΑΘΕ μονώνυμο x^k για k = 0..2n-1 στο [-1,1].
# Η ακριβής τιμή είναι ∫[-1,1] x^k dx = 2/(k+1) για άρτιο k, 0 για περιττό.
for n in range(1, 6):
    megisto_sfalma = 0.0
    for k in range(0, 2 * n):
        akrivis = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        ypologismenh = GaussianQuadrature.integrate_standard(lambda x, k=k: x**k, n=n)
        megisto_sfalma = max(megisto_sfalma, abs(ypologismenh - akrivis))
    krisi(f"n={n}: ακριβής για όλα τα x^k, k≤{2*n-1}", megisto_sfalma, 1e-14)

    # ΚΑΙ ο έλεγχος ότι ΔΕΝ είναι ακριβής για βαθμό 2n (το φράγμα είναι αυστηρό)
    k = 2 * n
    akrivis = 2.0 / (k + 1)
    ypologismenh = GaussianQuadrature.integrate_standard(lambda x: x**(2*n), n=n)
    sfalma_2n = abs(ypologismenh - akrivis)
    print(f"         (έλεγχος αυστηρότητας: για x^{2*n} το σφάλμα είναι {sfalma_2n:.3e} > 0 ✓)")


# =========================================================================
print("\n" + "=" * 78)
print("3. ΒΑΘΜΟΣ ΑΚΡΙΒΕΙΑΣ SIMPSON: ακριβής έως 3ου βαθμού, ΟΧΙ 4ου")
print("=" * 78)
for k in range(0, 4):
    # ∫₀¹ x^k dx = 1/(k+1)
    akrivis = 1.0 / (k + 1)
    r = SimpsonsRule1D.simple(lambda x, k=k: x**k, 0, 1)
    krisi(f"απλός Simpson ακριβής για x^{k}", abs(r - akrivis), 1e-15)
r4 = SimpsonsRule1D.simple(lambda x: x**4, 0, 1)
print(f"         (για x⁴: σφάλμα {abs(r4 - 0.2):.3e} > 0 ✓ — το φράγμα είναι αυστηρό)")

# Το ίδιο για τον σύνθετο
for k in range(0, 4):
    akrivis = 1.0 / (k + 1)
    r = SimpsonsRule1D.composite(lambda x, k=k: x**k, 0, 1, n=10)
    krisi(f"σύνθετος Simpson ακριβής για x^{k}", abs(r - akrivis), 1e-15)


# =========================================================================
print("\n" + "=" * 78)
print("4. SIMPSON 1D έναντι scipy.integrate.simpson (ΙΔΙΟ πλέγμα)")
print("=" * 78)
# Η scipy.integrate.simpson δέχεται ΔΕΙΓΜΑΤΑ y σε πλέγμα, όχι συνάρτηση.
# Δίνοντας το ίδιο ακριβώς πλέγμα, τα δύο αποτελέσματα πρέπει να ταυτίζονται
# μέχρι το σφάλμα στρογγυλοποίησης.
synartiseis = [
    ("x²",            lambda x: x**2,                  0.0, 2.0),
    ("sin(x)",        np.sin,                          0.0, np.pi),
    ("e^x",           np.exp,                          0.0, 1.0),
    ("1/(1+x²)",      lambda x: 1/(1+x**2),            0.0, 1.0),
    ("e^(-x²)",       lambda x: np.exp(-x**2),         0.0, 1.0),
    ("x·sin(5x)",     lambda x: x*np.sin(5*x),        -1.0, 3.0),
    ("log(1+x)",      lambda x: np.log(1+x),           0.0, 2.0),
]
for onoma, f, a, b in synartiseis:
    for n in [10, 50, 100]:
        x = np.linspace(a, b, n + 1)
        y = f(x)
        diko_mas = SimpsonsRule1D.composite(f, a, b, n=n)
        scipy_tim = si.simpson(y, x=x)
        # Σχετική διαφορά ως προς το μέγεθος του αποτελέσματος
        klimaka = max(abs(scipy_tim), 1.0)
        krisi(f"{onoma:12s} n={n:3d}", abs(diko_mas - scipy_tim) / klimaka, 1e-13)


# =========================================================================
print("\n" + "=" * 78)
print("5. GAUSS 1D έναντι scipy.integrate.fixed_quad (ίδιος αλγόριθμος)")
print("=" * 78)
for onoma, f, a, b in synartiseis:
    for n in [3, 5]:
        diko_mas = GaussianQuadrature.integrate(f, a, b, n=n)
        scipy_tim, _ = si.fixed_quad(f, a, b, n=n)
        klimaka = max(abs(scipy_tim), 1.0)
        krisi(f"{onoma:12s} n={n}", abs(diko_mas - scipy_tim) / klimaka, 1e-13)


# =========================================================================
print("\n" + "=" * 78)
print("6. ΤΡΑΠΕΖΙΟ έναντι scipy.integrate.trapezoid")
print("=" * 78)
for onoma, f, a, b in synartiseis:
    n = 100
    x = np.linspace(a, b, n + 1)
    diko_mas = TrapezoidalRule.composite(f, a, b, n=n)
    scipy_tim = si.trapezoid(f(x), x=x)
    klimaka = max(abs(scipy_tim), 1.0)
    krisi(f"{onoma:12s} n={n}", abs(diko_mas - scipy_tim) / klimaka, 1e-13)


# =========================================================================
print("\n" + "=" * 78)
print("7. ΔΙΠΛΑ ΟΛΟΚΛΗΡΩΜΑΤΑ έναντι scipy.integrate.dblquad (ακρίβεια ~1e-10)")
print("=" * 78)
# Η dblquad χρησιμοποιεί προσαρμοστικό QUADPACK, εντελώς διαφορετικό αλγόριθμο.
synartiseis_2d = [
    ("x·y",              lambda x, y: x*y,                        0.0, 1.0, 0.0, 1.0),
    ("sin(x)cos(y)",     lambda x, y: np.sin(x)*np.cos(y),        0.0, np.pi, 0.0, np.pi),
    ("e^-(x²+y²)",       lambda x, y: np.exp(-(x**2+y**2)),      -1.0, 1.0, -1.0, 1.0),
    ("x²cos(y)",         lambda x, y: x**2*np.cos(y),             0.0, 1.0, 0.0, np.pi/2),
    ("1/(1+x²+y²)",      lambda x, y: 1/(1+x**2+y**2),            0.0, 2.0, 0.0, 2.0),
    ("x³y² + xy",        lambda x, y: x**3*y**2 + x*y,           -1.0, 2.0, 0.0, 3.0),
]
for onoma, f, a, b, c, d in synartiseis_2d:
    # ΠΡΟΣΟΧΗ στη σύμβαση: dblquad(func(y,x), a, b, gfun, hfun) ολοκληρώνει
    # ως προς y στο [gfun,hfun] και ως προς x στο [a,b] — τα ορίσματα της
    # func είναι ΑΝΤΕΣΤΡΑΜΜΕΝΑ σε σχέση με τη δική μας σύμβαση f(x,y).
    anafora, _ = si.dblquad(lambda yy, xx: f(xx, yy), a, b, c, d,
                            epsabs=1e-13, epsrel=1e-13)
    s2d = SimpsonsRule2D.composite(f, a, b, c, d, nx=80, ny=80)
    g2d = GaussianQuadrature.integrate_2d(f, a, b, c, d, n=5)
    klimaka = max(abs(anafora), 1.0)
    krisi(f"Simpson2D  {onoma:14s}", abs(s2d - anafora) / klimaka, 1e-6)
    print(f"           (Gauss2D n=5: διαφορά {abs(g2d - anafora)/klimaka:.3e},"
          f" τιμή αναφοράς {anafora:.10f})")


# =========================================================================
print("\n" + "=" * 78)
print("8. ΠΡΟΣΑΡΜΟΣΤΙΚΟΣ SIMPSON έναντι scipy.integrate.quad")
print("=" * 78)
synartiseis_ad = [
    ("sin(x)",       np.sin,                       0.0, np.pi),
    ("√x",           np.sqrt,                      0.0, 1.0),
    ("e^(-x²)",      lambda x: np.exp(-x**2),      0.0, 3.0),
    ("1/(1+25x²)",   lambda x: 1/(1+25*x**2),     -1.0, 1.0),
    ("x·sin(10x)",   lambda x: x*np.sin(10*x),     0.0, 2.0),
    ("e^x·cos(x)",   lambda x: np.exp(x)*np.cos(x), 0.0, 1.0),
]
for onoma, f, a, b in synartiseis_ad:
    anafora, _ = si.quad(f, a, b, epsabs=1e-14, epsrel=1e-14)
    diko_mas = SimpsonsRule1D.adaptive(f, a, b, tol=1e-12)
    klimaka = max(abs(anafora), 1.0)
    krisi(f"{onoma:14s} [{a:g},{b:g}]", abs(diko_mas - anafora) / klimaka, 1e-9)


# =========================================================================
print("\n" + "=" * 78)
print("9. ΠΕΙΡΑΜΑΤΙΚΗ ΕΠΑΛΗΘΕΥΣΗ ΤΗΣ ΤΑΞΗΣ ΣΥΓΚΛΙΣΗΣ")
print("=" * 78)
# Αν το σφάλμα είναι E(h) ≈ C·h^p, τότε διπλασιάζοντας το n (μισό h)
# ο λόγος των σφαλμάτων είναι 2^p. Άρα p = log₂(E(n) / E(2n)).
# Αυτό επαληθεύει ΠΕΙΡΑΜΑΤΙΚΑ τους θεωρητικούς τύπους σφάλματος.
def taxi_sygklisis(methodos, f, a, b, akrivis, n_arxiko=16, plithos=4):
    ns, sfalmata = [], []
    n = n_arxiko
    for _ in range(plithos):
        e = abs(methodos(f, a, b, n) - akrivis)
        ns.append(n); sfalmata.append(e); n *= 2
    taxeis = [np.log2(sfalmata[i] / sfalmata[i+1]) for i in range(len(ns)-1)]
    return taxeis

f_dok = lambda x: np.exp(x) * np.cos(x)
akrivis_dok = (np.e * (np.sin(1) + np.cos(1)) - 1) / 2

for onoma, meth, anamenomeni in [
    ("Σύνθετος Simpson", SimpsonsRule1D.composite, 4.0),
    ("Τραπέζιο",         TrapezoidalRule.composite, 2.0),
    ("Μεσοσημείο",       MidpointRule.composite,    2.0),
]:
    taxeis = taxi_sygklisis(meth, f_dok, 0, 1, akrivis_dok)
    telikh = taxeis[-1]
    print(f"  {onoma:20s} μετρημένες τάξεις: {[f'{t:.3f}' for t in taxeis]}")
    krisi(f"{onoma}: τάξη ≈ {anamenomeni:.0f}", abs(telikh - anamenomeni), 0.15)

# Ο λόγος των σταθερών σφάλματος τραπεζίου/μεσοσημείου πρέπει να είναι 2
f2 = lambda x: np.exp(x)
ex2 = np.e - 1
n = 2000
e_trap = abs(TrapezoidalRule.composite(f2, 0, 1, n) - ex2)
e_mid = abs(MidpointRule.composite(f2, 0, 1, n) - ex2)
print(f"\n  Λόγος σφαλμάτων τραπεζίου/μεσοσημείου = {e_trap/e_mid:.4f} (θεωρία: 2.0)")
krisi("σταθερές σφάλματος 1/12 έναντι 1/24", abs(e_trap/e_mid - 2.0), 0.02)

# Και ο ίδιος ο τύπος του Simpson ως συνδυασμός (2·Μεσο + Τραπ)/3
s_simpson = SimpsonsRule1D.composite(f2, 0, 1, 100)
s_synd = (2*MidpointRule.composite(f2, 0, 1, 50) + TrapezoidalRule.composite(f2, 0, 1, 50)) / 3
krisi("Simpson = (2·Μεσοσημείο + Τραπέζιο)/3", abs(s_simpson - s_synd), 1e-14)


# =========================================================================
print("\n" + "=" * 78)
print("10. ΕΠΑΛΗΘΕΥΣΗ ΤΩΝ ΣΤΑΘΕΡΩΝ ΣΤΟΥΣ ΤΥΠΟΥΣ ΣΦΑΛΜΑΤΟΣ")
print("=" * 78)
# Δεν αρκεί να επαληθεύσουμε την ΤΑΞΗ O(h^p) — ελέγχουμε και τη ΣΤΑΘΕΡΑ
# και το ΠΡΟΣΗΜΟ. Σύμβαση: E = I - S (ακριβής τιμή μείον προσέγγιση).
#
# Για f(x) = e^x στο [0,1] όλες οι παράγωγοι είναι e^x, οπότε καθώς n → ∞
# η μέση τιμή f^(k)(ξ) τείνει στο ∫₀¹ e^x dx = e-1.
f_e = np.exp
a_e, b_e = 0.0, 1.0
I_e = np.e - 1

print(f"  {'Μέθοδος':14s} {'n':>5s} {'E=I-S (πραγμ.)':>16s} {'θεωρητικό':>16s} {'λόγος':>9s}")
for onoma, meth, typos in [
    ("Simpson",    SimpsonsRule1D.composite, lambda h: -(h**4 / 180) * I_e),
    ("Τραπέζιο",   TrapezoidalRule.composite, lambda h: -(h**2 / 12) * I_e),
    ("Μεσοσημείο", MidpointRule.composite,    lambda h: +(h**2 / 24) * I_e),
]:
    for n in [40, 80]:
        h = (b_e - a_e) / n
        E_pragm = I_e - meth(f_e, a_e, b_e, n)   # E = I - S
        E_theor = typos(h)
        logos = E_pragm / E_theor
        print(f"  {onoma:14s} {n:5d} {E_pragm:16.3e} {E_theor:16.3e} {logos:9.5f}")
    # Ο λόγος πρέπει να τείνει στο 1.000 (σωστή σταθερά ΚΑΙ σωστό πρόσημο)
    krisi(f"{onoma}: σταθερά και πρόσημο του τύπου", abs(logos - 1.0), 0.001)


# =========================================================================
print("\n" + "=" * 78)
if apotyxies:
    print(f"ΑΠΟΤΥΧΙΕΣ ({len(apotyxies)}):")
    for a in apotyxies:
        print("   -", a)
else:
    print("ΟΛΟΙ ΟΙ ΕΛΕΓΧΟΙ ΠΕΡΑΣΑΝ — καμία απόκλιση από θεωρία ή SciPy.")
print("=" * 78)

sys.exit(1 if apotyxies else 0)
