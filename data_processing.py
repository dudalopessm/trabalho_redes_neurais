import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import xarray as xr

OUTPUT_DIR1 = "cpc_precip"

ds_precip = xr.open_mfdataset(f"{OUTPUT_DIR1}/precip.*.nc", combine="by_coords")
ds_precip.precip.isel(time=0).plot()
plt.show()