# Crystal Lattices
In general, a 3-dimensional crystal is defined using a *Bravais lattice*. 
The Bravais lattice is defined by three primitive vectors $`\vec{a}_{1}`$, $`\vec{a}_{2}`$, and $`\vec{a}_{3}`$. 
For the 3-dimensional Bravais lattice, these three primitive vectors are used to tesselate $`\mathbb{R}^{3}`$:
```math
  \vec{r} = n_{1}\vec{a}_{1} + n_{2}\vec{a}_{2} + n_{3}\vec{a}_{3} \ : \ n_{1},n_{2},n_{3} \in \mathbb{Z}
```
In the simplest case, an atom may be placed directly on the Bravais points $\{\vec{r}\}$. 
In reality, it's extremely common for atoms to be at *non-Bravais* points, requiring basis positions $\{\vec{\delta}\}$. 
Then, the atomic position vector $\vec{r}\prime = \vec{r} + \vec{\delta}$. 

The Bravais lattice has a dual which is commonly referred to as the reciprocal-space or momentum-space lattice.
The primitive vectors for the reciprocal-space lattice are defined:
```math
    \begin{split}
        &\vec{b}_{1} = \frac{2\pi}{V}(\vec{b}_{2} \times \vec{b}_{1})\\
        &\vec{b}_{2} = \frac{2\pi}{V}(b_{3}\times\vec{b}_{1})\\
        &\vec{b}_{3} = \frac{2\pi}{V}(\vec{b}_{1}\times\vec{b}_{2})
    \end{split}
```
where $`V = \vec{b}_{1}\cdot(\vec{b}_{2}\times\vec{b}_{3})`$ is the volume of the unit cell.
The reciprocal-space lattice is tesselated:
```math
\vec{q} = h\vec{b}_{1} + k\vec{b}_{2} + \ell\vec{b}_{3}
```
where $[hk\ell]$ define the \text{Miller indices} of the wave-vector $\vec{q}$. 
The Miller indices define the spacing between planes in the reciprocal lattice:
```math
d = \frac{2\pi}{|\vec{q}_{hk\ell}|}
```
The reciprocal-space vectors are translationally symmetric such that $\vec{q} = \vec{q} + \vec{k}$ 
for $\vec{k} \in \{\vec{q}_{m}\}$ belonging to the set of reciprocal lattice points.
# Scattering Theory
The one-phonon dynamic structure factor is the spectral function corresponding to *inelastic* scattering 
events where the energy transfer $\Delta E \neq 0$. During the inelastic scattering, the neutron either 
loses energy (creates a phonon) or gains energy (destroys a phonon). The one-phonon contribution to the 
structure factor is defined:
```math
S_{\text{ph}}^{\pm}(\vec{q},\omega; \beta) = \sum_{\nu}\sum_{\vec{k}}A_{\nu}(\vec{q})\Delta_{\nu}^{\pm}(\vec{q},\vec{k})\left(n_{\nu}(\beta) + \frac{1 \pm 1}{2}\right)
```
where $A_{\nu}(\vec{q})$ is the scattering amplitude of the $\nu^{th}$ phonon mode induced by the momentum transfer $\mathbf{q}$. The scattering amplitude is related to the $\nu^{th}$ phonon mode polarization vector $\vec{\xi}_{\nu}$ such that $`A_{\nu}(\mathbf{q}) = (\vec{q}\cdot\vec{\xi}_{\nu})^{2}/\omega_{\nu}`$ 
for single-phonon scattering events.The Bose-Einstein distribution includes the thermal dependence such that $`n_{\nu}(\beta) = [e^{-\beta\hbar\omega_{\nu}} - 1]^{-1}`$ with $`\beta = (k_{b}T)^{-1}`$. The selection rules for scattering are enforced by,$`\Delta_{\nu}^{\pm}(\vec{q},\vec{k}) = \delta(\omega \pm \omega_{\nu})\delta(\mathbf{q} -\mathbf{k})`$ where $`\delta()`$ denotes a Dirac delta. 
