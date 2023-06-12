import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec

def SurFacePlot(x2, y2, z2, grid_x, grid_y, grid_z, zlabel, outlabel, PVModule, path, save=False, angle=30, s=10):
  cmap = plt.cm.get_cmap("jet", 20)
  fig = plt.figure(figsize=(16, 12), dpi=80)
  gs = gridspec.GridSpec(7, 4, width_ratios=[10, 10, 10, 1], height_ratios=[1, 10, 1, 10, 1, 10, 1], hspace=0.3, wspace=0.1)
  ax1 = plt.subplot(gs[:, 1:3], projection='3d'); 
  ax1.view_init(30, angle)
  ax2 = plt.subplot(gs[1, 0])
  ax3 = plt.subplot(gs[3, 0])
  ax4 = plt.subplot(gs[5, 0])
  axb = plt.subplot(gs[1:6, 3])
  ax2.grid(which='minor', alpha=0.5)
  ax3.grid(which='minor', alpha=0.5)
  ax4.grid(which='minor', alpha=0.5)
  
  norm = matplotlib.colors.Normalize(vmin=np.nanmin(z2), vmax=np.nanmax(z2))
  surf = ax1.plot_surface(grid_x, grid_y, grid_z, facecolors=cmap(norm(grid_z)), shade=False, antialiased=False, cstride=1, rstride=1, lw=0)
  m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  m.set_array([])
  cbar = fig.colorbar(m, cax=axb, extend='both', ticks=np.linspace(np.nanmin(z2), np.nanmax(z2), 6), format="%.2f")
  cbar.mappable.set_clim([np.nanmin(z2), np.nanmax(z2)])
  axb.set_title(zlabel)  
  
  ax1.scatter(x2, y2, z2, c="black",zorder=10)
  ax1.set_xlabel('Irradiance (W/m$^2$)')
  ax1.set_ylabel('Temperature (°C)')
  ax1.set_zlabel(zlabel)
  
  ax2.pcolor(grid_x, grid_y, grid_z, cmap=cmap, norm=norm, antialiased=False, lw=1)
  ax2.scatter(x2, y2, s=s, color='black')
  ax2.set_xlabel('Irradiance (W/m$^2$)')
  ax2.set_ylabel('Temperature (°C)')
  cbar = fig.colorbar(m, ax=ax2, extend='both', ticks=np.linspace(np.nanmin(z2), np.nanmax(z2), 6), format="%.2f")
  cbar.set_label(zlabel)

  norm = matplotlib.colors.Normalize(vmin=np.nanmin(y2), vmax=np.nanmax(y2))
  ax3.pcolor(grid_x, grid_z, grid_y, cmap=cmap, norm=norm, antialiased=False, lw=1)
  ax3.scatter(x2, z2, s=s, color='black')
  ax3.set_xlabel('Irradiance (W/m$^2$)')
  ax3.set_ylabel(zlabel)
  m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  m.set_array([])
  cbar = fig.colorbar(m, ax=ax3, extend='both', ticks=np.linspace(np.nanmin(y2), np.nanmax(y2), 6))
  cbar.set_label('Temperature (°C)')

  norm = matplotlib.colors.Normalize(vmin=np.nanmin(x2), vmax=np.nanmax(x2))
  ax4.pcolor(grid_y, grid_z, grid_x, cmap=cmap, norm=norm, antialiased=False, lw=1)
  ax4.scatter(y2, z2, s=s, color='black')
  ax4.set_xlabel('Temperature (°C)')
  ax4.set_ylabel(zlabel)
  ax1.set_title(PVModule)
  m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  m.set_array([])
  cbar = fig.colorbar(m, ax=ax4, extend='both', ticks=np.linspace(np.nanmin(x2), np.nanmax(x2), 6))
  cbar.set_label('Irradiance (W/m$^2$)')
  # if save:
  #   plt.savefig(normpath(path, outlabel), bbox_inches='tight')
  # plt.show()
