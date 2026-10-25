# Transform model cards

Every built-in transform has a packaged model card available through `medaugmentx.model_card`. These describe the implemented approximation and its input domain; no card establishes clinical validity.

## RandomAffine

**Input:** real 2D/3D; any intensity units. **Units:** degrees; dimensionless scale; fraction/voxels/mm translation.

`q = F(p-c)+c+t; physical F uses the grid metric`

active anatomy warp on a fixed patient grid; mask order 0.

**Limits:** Spline interpolation can overshoot; angles are in the image frame, not anatomical RAS axes.

References: [Source 1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html).

## RandomFlip

**Input:** real 2D/3D; task permits reversal. **Units:** array axis; probabilities.

`q[a] = size[a]-1-p[a]`

active anatomy reflection; optional explicit label permutations.

**Limits:** Array x is not automatically anatomical left-right; presets disable flips by default.

## AnatomicCrop

**Input:** real 2D/3D; optional integer mask. **Units:** positive voxel counts; foreground threshold in native units.

`q = p - crop_origin`

foreground means nonzero mask or image above threshold.

**Limits:** Biased sampling can change class prevalence; it does not identify anatomy.

## ElasticDeform

**Input:** real 2D/3D; integer labels. **Units:** alpha/sigma/cap in voxels or mm along grid axes.

`out[q] = in[q+d(q)]`

Gaussian-smoothed fields; sampled gradient limiting; conservative global box bounds.
Constant image padding uses nearest-boundary field extension for target inversion;
conservative boxes do not require corner inversion.

**Limits:** Discrete Jacobian checks do not prove global anatomy; conservative boxes can be loose.

References: [Source 1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html).

## CoarseDropout

**Input:** real 2D/3D. **Units:** hole extent as axis fraction; fill in image units.

`out[hole] = fill_value`

occlusion preserves labels unless fill_mask=True.

**Limits:** Does not simulate a particular detector or pathology.

## Resize

**Input:** real 2D/3D; registered targets. **Units:** target voxel counts.

`p = (q+0.5)*old/new-0.5`

full voxel extent preserved, including singleton axes; optional image antialias.

**Limits:** Resampling can remove small labels; spline overshoot is optional to clamp.

References: [Source 1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html).

## Pad

**Input:** real 2D/3D. **Units:** target voxel counts; fill in image units.

`q = p + padding_before`

sampling-grid affine is updated; mask padding is background 0.

**Limits:** Fill is not inferred from modality; choose CT air or normalized background explicitly.

## CenterCrop

**Input:** real 2D/3D. **Units:** target voxel counts.

`q = p - floor((old-crop)/2)`

sampling-grid affine is updated.

**Limits:** Targets may leave the frame; pruning is explicit.

## GaussianNoise

**Input:** real image in the declared native or normalized domain. **Units:** sigma in intensity units, or fraction of image standard deviation.

`out = image + N(0,sigma^2)`

independent Gaussian perturbations per voxel.

**Limits:** Approximation for CT noise; does not model dose, reconstruction kernel, or spatial correlations.

## RicianNoise

**Input:** nonnegative magnitude signal in the declared intensity domain. **Units:** per-complex-channel sigma in intensity units.

`out = hypot(image + N1, N2)`

independent equal-variance real/imaginary Gaussian channels.

**Limits:** Already noisy magnitude input is an approximation; multi-coil/reconstruction statistics can differ.

References: [Source 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2254141/), [Source 2](https://pubmed.ncbi.nlm.nih.gov/18440746/).

## GammaCorrection

**Input:** finite real image with a nonzero range. **Units:** positive dimensionless gamma.

`out = lo + range*normalize(image)^gamma`

per-image min-max normalization; optional inversion.

**Limits:** Changes calibrated intensity semantics; outliers affect the normalization.

## BiasField

**Input:** finite real image. **Units:** dimensionless log amplitude; coarse voxel grid.

`out = image * exp(clipped_interpolated_log_field)`

smooth multiplicative appearance field.

**Limits:** Not a B0/B1 or coil sensitivity simulation; large alpha can be unsuitable.

## WindowLevel

**Input:** native CT/HU or explicitly declared image units. **Units:** center/width in native units; perturbation fractions.

`out = clip(image, center-width/2, center+width/2)`

optional output normalization is explicit and updates domain metadata.

**Limits:** Display mapping, not acquisition dose/protocol simulation.

## BrightnessContrast

**Input:** native or normalized input as configured. **Units:** brightness in intensity units; dimensionless contrast.

`out = image*contrast + brightness`

scalar contrast is a fixed multiplier.

**Limits:** Does not simulate calibrated exposure or detector physics.

## GaussianBlur

**Input:** real 2D/3D. **Units:** sigma in voxels or per-axis mm.

`out = GaussianFilter(image, sigma)`

same sampled sigma; mask unchanged.

**Limits:** Separable grid-axis blur; mm on a sheared grid is not an isotropic RAS kernel.

## MedianBlur

**Input:** real 2D/3D. **Units:** positive odd voxel window size.

`out[q] = median(local_window)`

mask unchanged; cube/square neighborhood.

**Limits:** Large 3D windows are expensive; not a physical acquisition model.

## SimulateLowResolution

**Input:** real 2D/3D. **Units:** dimensionless downsample factor.

`out = upsample(downsample(antialias(image)))`

voxel-extent sampling; output grid and mask remain fixed.

**Limits:** Approximation to resolution loss, not scanner PSF or acquisition reconstruction.

## Sharpen

**Input:** real 2D/3D. **Units:** dimensionless alpha; sigma in voxels.

`out = image + alpha*(image-GaussianBlur(image))`

unsharp-mask appearance perturbation.

**Limits:** Can overshoot native range; not a calibrated vendor reconstruction kernel.

## CLAHEContrast

**Input:** finite real image. **Units:** relative histogram clip limit; tile counts; bins.

`out = blended_local_clipped_histogram_CDF(image)`

3D is processed slice-by-slice on array axis 0.

**Limits:** Implementation-specific CLAHE; may introduce inter-slice intensity differences.

## HistogramMatch

**Input:** finite image; reference training distribution. **Units:** dimensionless blend; quantile count.

`out = (1-blend)*image + blend*reference_CDF_inverse(source_CDF)`

reference quantiles saved instead of reference image.

**Limits:** Fit references on training data only; histogram matching does not preserve HU calibration.

## SlabShift

**Input:** 3D stack. **Units:** signed slices; fill in intensity units.

`q = p + (shift,0,0)`

active anatomy shift on fixed grid; mask and targets shifted.

**Limits:** Image-domain reconstruction-center approximation; targets may leave the frame.

## LimitedAngleBlur

**Input:** 3D stack. **Units:** arc degrees; sigma in slices.

`sigma = base_sigma*reference_arc/arc`

Gaussian blur only on array slice axis.

**Limits:** Inverse-angle relationship is heuristic; no projection/reconstruction model.

## SliceDropout

**Input:** 3D stack. **Units:** integer slice count; fill in intensity units.

`out[selected_slices] = fill`

mask remains unless affect_mask=True.

**Limits:** Robustness corruption; not a validated missing-frame acquisition simulator.

## AnisotropicElastic

**Input:** 3D stack. **Units:** per-axis alpha/sigma in voxels or mm.

`out[q] = in[q+d(q)]`

delegates to bounded ElasticDeform in zyx order.

**Limits:** Defaults require dataset review; conservative box bounds can be loose.

## CompressionVariation

**Input:** 3D stack. **Units:** dimensionless in-plane scale.

`q[a] = center[a] + scale*(p[a]-center[a])`

fixed output grid; labels use nearest neighbor.

**Limits:** Appearance approximation along one axis; not a mechanical breast-compression model.

## ReconStreak

**Input:** 3D stack. **Units:** dimensionless amplitude/decay; voxel shifts.

`out = image + sum(weight*shifted_neighbor_planes)`

neighbor replicas with in-plane parallax.

**Limits:** Heuristic reconstruction appearance; not calibrated to scanner geometry.

## GhostingArtifact

**Input:** real MRI-like magnitude image. **Units:** relative replica amplitude; shift in voxels.

`out = image + sum(amplitude*rolled_image)`

periodic shifted image-domain replicas.

**Limits:** No acquisition timing or complex phase model; not all MRI ghosting.

## KSpaceDropout

**Input:** real reconstructed image; MRI approximation. **Units:** fraction of phase-encode lines.

`out = abs(IFFT(mask * FFT(image)))`

independent 2D FFT corruption per slice.

**Limits:** Measured complex k-space is unavailable; DC line removal can drastically change mean.

## MRIMotion

**Input:** real MRI-like reconstructed image. **Units:** degrees; voxel translations; state count.

`out = mean(original, moved_image_states)`

same in-plane motion states for every slice.

**Limits:** Image averaging differs from motion during actual k-space acquisition.

## BeamHardening

**Input:** native or normalized CT-like image. **Units:** alpha as image-range fraction; dimensionless power.

`out = image - alpha*image_range*radial_bowl`

fixed image-centered cupping profile.

**Limits:** No spectrum/material/ray tracing; can darken background or off-center anatomy.

## MetalStreak

**Input:** native or normalized CT-like image. **Units:** range-relative amplitude; angular lobes; radial falloff fraction.

`out = image + range*amplitude*starburst_field`

random source locations shared across slices.

**Limits:** Not metal segmentation, sinogram corruption, or a reconstruction simulation.

## ScatterSimulation

**Input:** real radiograph-like image. **Units:** relative fraction; sigma in voxels.

`out = image + fraction*GaussianBlur(image)`

smooth image-domain additive background.

**Limits:** No photon transport or detector response; sigma affects all axes on 3D input.

## GridArtifact

**Input:** real radiograph-like image. **Units:** relative amplitude; cycles per pixel.

`out = image*(1+amplitude*sin(2*pi*frequency*x+phase))`

sinusoidal stationary stripes.

**Limits:** No grid geometry/motion model; parameters near Nyquist can alias.

## NormalizeIntensity

**Input:** finite real image. **Units:** percentiles in [0,100].

`out = (clip(image,lo,hi)-lo)/(hi-lo)`

deterministic; constant images become zero.

**Limits:** Changes intensity calibration; use before normalized-domain effects, never to preserve HU.
