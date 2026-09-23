(core-match-ref)=
# Match-reference functionality

Match-reference functionalities lie at the core of GeoUtils' focus to provide accessible and intuitive tools for end-user geospatial analysis.

## What is match-reference ?

Match-reference means to **match** the georeferencing of a dataset to that of another **reference** dataset.

## Why the need for match-reference?

End-users of geospatial tools largely focus on analyzing their data. To this end, they first **need to reconcile data sources**, often provided **in different
projections, resolutions, and bounds**. This is generally performed by matching all dataset to a certain reference.

Such functionalities are not always available from lower-level packages, in order to keep geoinformatics flexible for more diverse uses. This
**breaks the principle of least knowledge**, however, for which a user should not need to understand more information than just what is required to run an
operation.

**GeoUtils allows for match-reference for nearly all geospatial handling operations**, with consistent behaviour across functionalities for intuitive use.

## Matching with a {class}`~geoutils.Raster` or a {class}`~geoutils.Vector` reference

The rules of using match-reference with {class}`Rasters<geoutils.Raster>` or {class}`Vectors<geoutils.Vector>` are always the same:

 - If the **reference** passed is a {class}`~geoutils.Vector`, it can enforce a matching of its {attr}`~VectorBase.bbox` and/or of its {attr}`~VectorBase.crs` (its only two
   georeferencing attributes),
 - If the **reference** is a {class}`~geoutils.Raster`, it can also enforce a matching of any aspect of its {attr}`~RasterBase.transform` (i.e, its
   {attr}`~RasterBase.res`, {attr}`~RasterBase.bbox` or {attr}`~RasterBase.shape`) and/or of its {attr}`~RasterBase.crs`.

Which of these attributes are eventually used to enforce the matching **depends entirely on the nature of the operation**, which are listed below.

## Geospatial handling rules for match-reference

Geospatial handling methods all support match-reference, and **always enforce the same georeferencing attributes** for either {class}`~geoutils.Raster`
or {class}`~geoutils.Vector`:

```{list-table}
   :widths: 30 30 30
   :header-rows: 1

   * - **Operation**
     - Enforced on {class}`~geoutils.Raster`
     - Enforced on {class}`~geoutils.Vector`
   * - {meth}`~RasterBase.reproject`
     - {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
     - {attr}`~VectorBase.bbox`<sup>1</sup> and {attr}`~VectorBase.crs`
   * - {meth}`~RasterBase.crop`
     - {attr}`~VectorBase.bbox`
     - {attr}`~RasterBase.bbox`

```

<sup>1</sup>Because a {class}`~geoutils.Vector` only possesses the {attr}`~VectorBase.bbox` attribute of a {class}`~geoutils.Raster`'s {attr}`~RasterBase.transform`.


## Other operations supporting match-reference

There are **other geospatial operation that also support match-reference arguments**. Unlike the geospatial handling methods described above, these do not aim
at modifying the georeferencing of {class}`Rasters<geoutils.Raster>` or {class}`Vectors<geoutils.Vector>`. Instead, they simply require the georeferencing metadata.

### From vector to raster

The {meth}`~VectorBase.rasterize` operation to convert from {class}`~geoutils.Vector` to {class}`~geoutils.Raster` accepts a {class}`~geoutils.Raster` to define the
grid and georeferencing. The behaviour is similar for {meth}`~VectorBase.create_mask`, that directly relies on {meth}`~VectorBase.rasterize` to
rasterize directly into a boolean {class}`~geoutils.Raster`.

In addition, the {meth}`~VectorBase.proximity` operation to compute proximity distances from the vector also relies on a
{meth}`~VectorBase.rasterize`, and therefore also accepts a {class}`~geoutils.Raster` as reference.

Therefore, the behaviour is consistent for all {class}`~geoutils.Vector` methods that can be passed a {class}`~geoutils.Raster`:

```{list-table}
   :widths: 50 50
   :header-rows: 1

   * - **Operation on {class}`~geoutils.Vector`**
     - **Behaviour**
   * - {meth}`~VectorBase.rasterize`
     - Gridding with {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
   * - {meth}`~VectorBase.create_mask`
     - Gridding with {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
   * - {meth}`~VectorBase.proximity`
     - Gridding with {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
```

### And inversely

However, in the case of {class}`~geoutils.Raster` methods that yield a {class}`~geoutils.Vector` or {class}`~geoutils.Raster`, a reference is rarely needed.
This is because this reference is derived directly from the input {class}`~geoutils.Raster` itself, harnessing the object-based structure of GeoUtils.

The user can always {meth}`~VectorBase.crop` or {meth}`~VectorBase.reproject` the output afterwards, if desired.

```{list-table}
   :widths: 50 50
   :header-rows: 1

   * - **Operation on {class}`~geoutils.Raster`**
     - **Behaviour**
   * - {meth}`~RasterBase.polygonize`
     - Using `.self` ({class}`~geoutils.Raster`) as reference for {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
   * - {meth}`~RasterBase.proximity`
     - Using `.self` ({class}`~geoutils.Raster`) as reference for {attr}`~RasterBase.transform` and {attr}`~RasterBase.crs`
```
