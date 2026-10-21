// Start Xarray attributes collapsed so long CRS descriptions do not fill the page
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".xr-section-item").forEach((section) => {
        const attributes = section.querySelector(":scope > .xr-section-details > .xr-attrs");
        const toggle = section.querySelector(":scope > .xr-section-summary-in");
        if (attributes && toggle) {
            toggle.checked = false;
        }
    });
});
