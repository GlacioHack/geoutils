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

 - If the **reference** passed is a {class}`~geoutils.Vector`, it can enforce a matching of its {attr}`~geoutils.Vector.bounds` and/or of its {attr}`~geoutils.Vector.crs` (its only two
   georeferencing attributes),
 - If the **reference** is a {class}`~geoutils.Raster`, it can also enforce a matching of any aspect of its {attr}`~geoutils.Raster.transform` (i.e, its
   {attr}`~geoutils.Raster.res`, {attr}`~geoutils.Raster.bounds` or {attr}`~geoutils.Raster.shape`) and/or of its {attr}`~geoutils.Raster.crs`.

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
   * - {func}`~geoutils.Raster.reproject`
     - {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
     - {attr}`~geoutils.Vector.bounds`<sup>1</sup> and {attr}`~geoutils.Vector.crs`
   * - {func}`~geoutils.Raster.crop`
     - {attr}`~geoutils.Vector.bounds`
     - {attr}`~geoutils.Raster.bounds`

```

<sup>1</sup>Because a {class}`~geoutils.Vector` only possesses the {attr}`~geoutils.Vector.bounds` attribute of a {class}`~geoutils.Raster`'s {attr}`~geoutils.Raster.transform`.


## Other operations supporting match-reference

There are **other geospatial operation that also support match-reference arguments**. Unlike the geospatial handling methods described above, these do not aim
at modifying the georeferencing of {class}`Rasters<geoutils.Raster>` or {class}`Vectors<geoutils.Vector>`. Instead, they simply require the georeferencing metadata.

### From vector to raster

The {func}`~geoutils.Vector.rasterize` operation to convert from {class}`~geoutils.Vector` to {class}`~geoutils.Raster` accepts a {class}`~geoutils.Raster` to define the
grid and georeferencing. The behaviour is similar for {func}`~geoutils.Vector.create_mask`, that directly relies on {func}`~geoutils.Vector.rasterize` to
rasterize directly into a boolean {class}`~geoutils.Raster`.

In addition, the {func}`~geoutils.Vector.proximity` operation to compute proximity distances from the vector also relies on a
{func}`~geoutils.Vector.rasterize`, and therefore also accepts a {class}`~geoutils.Raster` as reference.

Therefore, the behaviour is consistent for all {class}`~geoutils.Vector` methods that can be passed a {class}`~geoutils.Raster`:

```{list-table}
   :widths: 50 50
   :header-rows: 1

   * - **Operation on {class}`~geoutils.Vector`**
     - **Behaviour**
   * - {func}`~geoutils.Vector.rasterize`
     - Gridding with {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
   * - {func}`~geoutils.Vector.create_mask`
     - Gridding with {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
   * - {func}`~geoutils.Vector.proximity`
     - Gridding with {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
```

### And inversely

However, in the case of {class}`~geoutils.Raster` methods that yield a {class}`~geoutils.Vector` or {class}`~geoutils.Raster`, a reference is rarely needed.
This is because this reference is derived directly from the input {class}`~geoutils.Raster` itself, harnessing the object-based structure of GeoUtils.

The user can always {func}`~geoutils.Vector.crop` or {func}`~geoutils.Vector.reproject` the output afterwards, if desired.

```{list-table}
   :widths: 50 50
   :header-rows: 1

   * - **Operation on {class}`~geoutils.Raster`**
     - **Behaviour**
   * - {func}`~geoutils.Raster.polygonize`
     - Using `.self` ({class}`~geoutils.Raster`) as reference for {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
   * - {func}`~geoutils.Raster.proximity`
     - Using `.self` ({class}`~geoutils.Raster`) as reference for {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.crs`
```
