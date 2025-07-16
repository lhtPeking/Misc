# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
#matplotlib.use("cairo") # does not support rasterization of lines, but allow embedding subset of fonts

import os
import sys
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' 
import pylab as pl

python_file_path = os.path.dirname(os.path.abspath(__file__))

from matplotlib import rcParams
from matplotlib import font_manager

class Figure():
    def __init__(self, title=None, lc="black", lw=1.1, pt='o', ps=None, pc='black', errorbar_area = True,
                 auto_panel_letters = True, textcolor = 'black',
                 fig_width = 21.59, fig_heigth = 27.94, plot_width=4.5, plot_height=3.75,
                 fname = os.path.join(python_file_path, 'Arial.ttf'), fontsize=9, # Arial Unicode.ttf for very special symbols (but large)
                 fname2 = os.path.join(python_file_path, 'Arial Bold.ttf'), fontsize2=14,
                 fname3 = os.path.join(python_file_path, 'Arial Bold.ttf'), fontsize3=16,
                 dashes=(2,2)):

        # embed font (in matplotlib only full font embed possible, still)
        #rcParams['pdf.fonttype'] = 3
        #rcParams['ps.fonttype'] = 3
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        self.font = font_manager.FontProperties(fname = fname, size=fontsize)
        self.font2 = font_manager.FontProperties(fname = fname2, size=fontsize2)
        self.font3 = font_manager.FontProperties(fname=fname3, size=fontsize3)
        self.fontsize2 = fontsize2
        self.fontsize = fontsize

        self.plot_width = float(plot_width)
        self.plot_height = float(plot_height)
        self.lc = lc
        self.lw = lw
        self.ps = ps
        self.dashes = dashes
        self.pt = pt
        self.pc = pc
        self.textcolor = textcolor
        self.auto_panel_letters = auto_panel_letters
        self.errorbar_area = errorbar_area

        
        if self.auto_panel_letters:
            self.autonum = 97
            
        self.fig_width_cm = float(fig_width)                      # A4 page 
        self.fig_height_cm = float(fig_heigth)
        
        inches_per_cm = 1 / 2.54              # Convert cm to inches 
        fig_width_inch = self.fig_width_cm * inches_per_cm # width in inches 
        fig_height_inch = self.fig_height_cm * inches_per_cm       # height in inches 
        fig_size_inch = [fig_width_inch, fig_height_inch] 
        
        self.fig = pl.figure(num=None, figsize=fig_size_inch, facecolor='none', edgecolor='none')

        if title is not None:
            pl.figtext(0.5, 0.95, title, ha='center', va='center', fontproperties=self.font3, color=self.textcolor)
    
    def savepdf(self, path, open_pdf=False, add_python_file=None, tight=False):

        path_ =  path.replace("\n", "_")

        try:

            if tight:

                self.fig.savefig(path_+".pdf", facecolor='none', edgecolor='none', bbox_inches='tight', dpi=600)
            else:

                self.fig.savefig(path_+".pdf", facecolor='none', edgecolor='none', dpi=600)

        except Exception as e:
            print("Could not save ", path_, ".pdf", e)

        ## close the figure
        pl.close()

        if add_python_file is not None and sys.platform.startswith('darwin'):

            os.system("/Users/arminbahl/Desktop/text2pdf -s6 -c150 %s > '/tmp/sourcecode.pdf'"%add_python_file)
            os.system("/Library/TeX/texbin/pdfjam --papersize '{%fcm,%fcm}' -o '%s.pdf' '%s.pdf' '/tmp/sourcecode.pdf'"%(self.fig_width_cm, self.fig_height_cm, path_, path_))
        
        if open_pdf:
            if sys.platform.startswith('darwin'):
                os.system("open '%s.pdf'" % path_)
            else:
                os.startfile("%s.pdf" % path_)

    def savepng(self, path, tight=False):
        path_ =  path.replace("\n", "_")
        if tight:
            self.fig.savefig(path_+".png", facecolor='none', edgecolor='none', bbox_inches='tight', transparent=True, dpi=400)
        else:
            self.fig.savefig(path_+".png", facecolor='none', edgecolor='none', transparent=True, dpi=400)
        return

    def add_text(self, x, y, text, font = None, rotation=0, ha = 'center'):
        pl.figtext(x/self.fig_width_cm, y/self.fig_height_cm, text, fontproperties=self.font if font is None else font, ha = ha, ma=ha, va='center', color=self.textcolor, rotation=rotation)


class PolarPlot():
    def __init__(self, fig, xpos, ypos, plot_width=None, plot_height=None, dashes=None, lc=None, lw=None,  pt=None, ps=None, pc=None,
                 yticklabels = None, xticklabels_rotation = 0, yticklabels_rotation=0,
                 ymin=None, ymax=None):
        self.fig = fig
        self.xpos = float(xpos)
        self.ypos = float(ypos)
        self.plot_width = plot_width
        self.plot_height = plot_height


        self.lc = lc
        self.lw = lw
        self.dashes = dashes
        self.pt = pt
        self.ps = ps
        self.pc = pc

        self.ymin = ymin
        self.ymax = ymax

        if self.plot_width is None:
            self.plot_width = fig.plot_width
        else:
            self.plot_width = float(self.plot_width)
        if self.plot_height is None:
            self.plot_height = fig.plot_height
        else:
            self.plot_height = float(self.plot_height)

        if self.lc is None:
            self.lc = fig.lc
        if self.lw is None:
            self.lw = fig.lw
        if self.ps is None:
            self.ps = fig.ps
        if self.pt is None:
            self.pt = fig.pt

        if self.pc is None:
            self.pc = fig.pc

        if self.dashes is None:
            self.dashes = fig.dashes

        self.ax = self.fig.fig.add_axes([self.xpos/self.fig.fig_width_cm, self.ypos/self.fig.fig_height_cm,
                                         self.plot_width/self.fig.fig_width_cm, self.plot_height/self.fig.fig_height_cm], polar=True)

        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)

        #if xticks is not None:

        xticks = np.arange(0, 360, 45)
        self.ax.set_xticks(xticks*np.pi/180.)
        self.ax.set_thetagrids(xticks, frac=1.2)

        xticks_ = xticks.copy()
        ind = np.where(xticks_>180)
        xticks_[ind] = xticks_[ind] - 360
        xticklabels = [str(lbl) for lbl in xticks_]

        for i in range(len(xticklabels)):
            xticklabels[i] = xticklabels[i].replace("-",  u'–')

            if xticklabels_rotation is 0:
                self.ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation, horizontalalignment='center', fontproperties=self.fig.font, color=self.fig.textcolor)
            else:
                self.ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation, horizontalalignment='right', fontproperties=self.fig.font, color=self.fig.textcolor)

        #else:
            #self.ax.spines['bottom'].set_visible(False)
        #    self.ax.get_xaxis().set_ticks([])

        #if yticks is not None:
        #    self.ax.set_yticks(yticks)

        #    if yticklabels is None:
        #        yticklabels = [str(lbl) for lbl in yticks]

        #if yticklabels is not None:
        #    for i in range(len(yticklabels)):
        #        yticklabels[i] = yticklabels[i].replace("-",  u'–')

        #    self.ax.set_yticklabels(yticklabels, rotation=yticklabels_rotation, horizontalalignment='right', fontproperties=self.fig.font, color=self.fig.textcolor)
        #else:
        #    #self.ax.spines['left'].set_visible(False)
        #self.ax.get_yaxis().set_ticks([])

        #if xmin is not None:

        self.ax.set_xlim([0, xticks[-1]])#, ymin, ymax])

        if ymin is not None:
            self.ax.set_ylim([ymin, ymax])#, ymin, ymax])


class Plot():
    def __init__(self, fig, xpos, ypos, num=None, xmin=None, xmax=None, ymin=None, ymax=None, xl=None, yl=None,
                 title=None, xticks=None, yticks=None, xlog = False,
                 xticklabels = None, yticklabels = None, xticklabels_rotation = 0, yticklabels_rotation=0, ylog=False,
                 plot_width=None, plot_height=None, dashes=None, lc=None, lw=None,  pt=None, ps=None, pc = None, errorbar_area = None,
                 legend_xpos = None, legend_ypos = None,
                 hlines = None,
                 vlines = None,
                 hspans = None,
                 vspans = None,
                 show_colormap = False, colormap = None, zmin = None, zmax = None, zticks = None, zticklabels = None,
                 zticklabels_rotation = 0, zl = None):


        self.fig = fig
        self.xpos = float(xpos)
        self.ypos = float(ypos)
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.lc = lc
        self.lw = lw
        self.dashes = dashes
        self.pt = pt
        self.ps = ps
        self.pc = pc

        self.show_colormap = show_colormap
        self.errorbar_area = errorbar_area

        if fig.auto_panel_letters is True and num is not '':
            if num is None:
                num = chr(fig.autonum)
            else:
                fig.autonum = ord(num)
            
            fig.autonum +=1
            
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.zticks = zticks
        self.zticklabels = zticklabels
        self.zticklabels_rotation = zticklabels_rotation
        self.colormap=colormap
        self.zl = zl
        self.legend_xpos = legend_xpos
        self.legend_ypos = legend_ypos
        
        if self.plot_width is None:
            self.plot_width = fig.plot_width
        else:
            self.plot_width = float(self.plot_width)
        if self.plot_height is None:
            self.plot_height = fig.plot_height
        else:
            self.plot_height = float(self.plot_height)    
            
        if self.lc is None:
            self.lc = fig.lc
        if self.lw is None:
            self.lw = fig.lw
        if self.ps is None:
            self.ps = fig.ps
        if self.pt is None:
            self.pt = fig.pt
        if self.pc is None:
            self.pc = fig.pc

        if self.dashes is None:
            self.dashes = fig.dashes

        if self.errorbar_area is None:
            self.errorbar_area = fig.errorbar_area

        if self.legend_xpos is None:
            self.legend_xpos = self.xpos + self.plot_width
        if self.legend_ypos is None:
            self.legend_ypos = self.ypos + self.plot_height
            
                
        self.ax = self.fig.fig.add_axes([self.xpos/self.fig.fig_width_cm, self.ypos/self.fig.fig_height_cm,
                                         self.plot_width/self.fig.fig_width_cm, self.plot_height/self.fig.fig_height_cm])

        #self.ax.set_facecolor("none")

        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')

        self.ax.spines['left'].set_linewidth(self.lw/2.)
        self.ax.spines['bottom'].set_linewidth(self.lw/2.)
        self.ax.spines['left'].set_color(self.fig.textcolor)
        self.ax.spines['bottom'].set_color(self.fig.textcolor)
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.tick_params('both', width=self.lw/2., which='major', tickdir="out", color=self.fig.textcolor)
        self.ax.tick_params('both', width=self.lw/2., which='minor', tickdir="out", color=self.fig.textcolor)

        #self.ax.set_autoscalex_on(False)
        if xmin is not None:
            self.ax.set_xlim([xmin, xmax])#, ymin, ymax])
        if ymin is not None:
            self.ax.set_ylim([ymin, ymax])#, ymin, ymax])

        if xl is not None:
            self.ax.set_xlabel(xl, horizontalalignment='center', fontproperties=self.fig.font, color=self.fig.textcolor)
            self.ax.xaxis.set_label_coords(0.5, -1./self.plot_height)
        if yl is not None:
            self.ax.set_ylabel(yl, verticalalignment='center', horizontalalignment='center',
                               fontproperties=self.fig.font, color=self.fig.textcolor)
            if '\n' in yl:
                self.ax.yaxis.set_label_coords(-1.3/self.plot_width*self.fig.fontsize/9., 0.5)
            else:
                self.ax.yaxis.set_label_coords(-1.3/self.plot_width*self.fig.fontsize/9., 0.5)

        if xlog is True:
            self.ax.set_xscale("log")

        if ylog is True:
            self.ax.set_yscale("log")


        if xticks is not None:
            self.ax.set_xticks(xticks)

            if xticklabels is None:
                xticklabels = [str(lbl) for lbl in xticks]

        if xticklabels is not None:
            for i in range(len(xticklabels)):
                xticklabels[i] = xticklabels[i].replace("-",  '–')

            if xticklabels_rotation is 0:
                self.ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation, horizontalalignment='center', fontproperties=self.fig.font, color=self.fig.textcolor)
            else:
                self.ax.set_xticklabels(xticklabels, rotation=xticklabels_rotation, horizontalalignment='right', fontproperties=self.fig.font, color=self.fig.textcolor)
        else:
            self.ax.spines['bottom'].set_visible(False)
            self.ax.tick_params(axis='x',which='minor', bottom='off')
            self.ax.tick_params(axis='x',which='major', bottom='off')
            self.ax.get_xaxis().set_ticks([])

        if yticks is not None:
            self.ax.set_yticks(yticks)

            if yticklabels is None:
                yticklabels = [str(lbl) for lbl in yticks]

        if yticklabels is not None:
            for i in range(len(yticklabels)):
                yticklabels[i] = yticklabels[i].replace("-",  '–')

            self.ax.set_yticklabels(yticklabels, rotation=yticklabels_rotation, horizontalalignment='right', fontproperties=self.fig.font, color=self.fig.textcolor)
        else:
            self.ax.spines['left'].set_visible(False)
            self.ax.tick_params(axis='y',which='minor', left='off')
            self.ax.tick_params(axis='y',which='major', left='off')
            self.ax.get_yaxis().set_ticks([])

        if hlines is not None:
            for hline in hlines:
                self.ax.axhline(hline, linewidth=0.25*self.lw, color=self.fig.textcolor, dashes=self.fig.dashes, solid_capstyle="round", dash_capstyle="round", zorder=0)
        if vlines is not  None:
            for vline in vlines:
                self.ax.axvline(vline, linewidth=0.25*self.lw, color=self.fig.textcolor, dashes=self.fig.dashes, solid_capstyle="round", dash_capstyle="round", zorder=0)
        if vspans is not  None:
            for vspan in vspans:
                self.ax.axvspan(vspan[0], vspan[1], lw = 0, edgecolor='none', facecolor=vspan[2],zorder=0, alpha=vspan[3])
        if hspans is not  None:
            for hspan in hspans:
                self.ax.axhspan(hspan[0], hspan[1], lw = 0, edgecolor='none', facecolor=hspan[2],zorder=0, alpha=hspan[3])

        if title is not  None:
            self.ax.set_title(title, fontproperties=self.fig.font, color=self.fig.textcolor)
        if num is not None:
            if self.ax.spines['left'].get_visible():
                pl.figtext((self.xpos-1.8*self.fig.fontsize/9.)/self.fig.fig_width_cm, (self.ypos+self.plot_height+0.5)/self.fig.fig_height_cm, num, fontproperties=self.fig.font2, weight='bold', fontsize = self.fig.fontsize2, ha = 'center', va='center', color=self.fig.textcolor)
            else:
                pl.figtext((self.xpos-0.3*self.fig.fontsize/9.)/self.fig.fig_width_cm, (self.ypos+self.plot_height+0.5)/self.fig.fig_height_cm, num, fontproperties=self.fig.font2, weight='bold', fontsize = self.fig.fontsize2, ha = 'center', va='center', color=self.fig.textcolor)
            
        if self.show_colormap is True:
                    
            cbar_ax = fig.fig.add_axes([(self.xpos+self.plot_width+self.plot_width/20.)/self.fig.fig_width_cm, self.ypos/self.fig.fig_height_cm, (self.plot_width/20.)/self.fig.fig_width_cm, self.plot_height/self.fig.fig_height_cm])
            cbar_ax.set_facecolor("none")
            cbar_ax2 = cbar_ax.twinx()
            cbar_ax2.tick_params('both', width=self.lw / 2., which='major', tickdir="out")

            cbar_ax2.imshow(np.c_[np.linspace(self.zmin, self.zmax, 500)], extent=(0, 1, self.zmin, self.zmax), aspect='auto', origin='lower', cmap=pl.get_cmap(self.colormap))
            cbar_ax.yaxis.set_ticks([])
            cbar_ax.xaxis.set_ticks([])
            
            if self.zticks is not None:
                cbar_ax2.set_yticks(self.zticks)

                if self.zticklabels is None:
                    self.zticklabels = [str(lbl) for lbl in self.zticks]

                for i in range(len(self.zticklabels)):
                    self.zticklabels[i] = self.zticklabels[i].replace("-", '–')
                cbar_ax2.set_yticklabels(self.zticklabels, rotation=zticklabels_rotation, horizontalalignment='left', fontproperties=self.fig.font, color=self.fig.textcolor)#self.fig.textcolor)
            else:
                print("need define zticks...")

            if zl is not None:
                cbar_ax2.set_ylabel(zl, fontproperties=self.fig.font, color=self.fig.textcolor)

class Line():
    def __init__(self, ax, x=None, y=None, yerr=None, xerr = None, lc=None, lw=None, dashes=None, pt = None, pc = None, ps=None, label=None, errorbar_area = None, rasterized=False, zorder=0):
        self.lc = lc
        self.lw = lw
        self.ps = ps
        self.dashes = dashes
        self.pt = pt
        self.pc = pc
        self.errorbar_area = errorbar_area

        # rasterization does not work with cairo
        if lc is None:
            self.lc = ax.lc
        
        if lw is None:
            self.lw = ax.lw
        if dashes is None:
            self.dashes = ax.dashes
        if pt is None:
            self.pt = ax.pt
                
        if ps is None:
            self.ps = ax.ps

        if pc is None:
            self.pc = ax.pc

        if errorbar_area is None:
            self.errorbar_area = ax.errorbar_area
        
        x = np.array(x)
        y = np.array(y)

        if yerr is not None:
            yerr = np.array(yerr)
            if self.errorbar_area == False:
                ax.ax.errorbar(x, y, yerr=yerr, elinewidth=1,ecolor=self.lc, fmt='none', capsize=2, mew=1, solid_capstyle='round', solid_joinstyle='round', zorder=zorder)
            else:
                ax.ax.fill_between(x, y-yerr, y+yerr, lw=0, edgecolor='none', facecolor=self.lc, alpha=0.2, zorder=zorder)
        
        if xerr is not None:
            xerr = np.array(xerr)
            if self.errorbar_area == False:
                ax.ax.errorbar(x, y, xerr=yerr, elinewidth=1,ecolor=self.lc, fmt='none', capsize=2, mew=1, solid_capstyle='round', solid_joinstyle='round', zorder=zorder)
            else:
                ax.ax.fill_betweenx(y, x-xerr, x+xerr, lw=0, edgecolor='none', facecolor=self.lc, alpha=0.2, zorder=zorder)

        if self.ps is not None:
            if dashes is not None:

                ax.ax.plot(x, y, color = self.lc, lw = self.lw, dashes=self.dashes, dash_capstyle="round",  dash_joinstyle="round", marker=self.pt, markersize=self.ps, markeredgewidth=self.ps/10., markerfacecolor=self.pc, markeredgecolor = self.lc, label=label, rasterized=rasterized, zorder=zorder)
            else:
                ax.ax.plot(x, y, color = self.lc, lw = self.lw, solid_capstyle="round", solid_joinstyle="round", marker=self.pt, markersize=self.ps, markeredgewidth=self.ps/10., markerfacecolor=self.pc, markeredgecolor = self.lc, label=label, rasterized=rasterized, zorder=zorder)
            
        else:
            
            if dashes is not None:
                ax.ax.plot(x, y, color = self.lc, lw = self.lw, dashes=self.dashes, dash_capstyle="round", label=label, rasterized=rasterized, zorder=zorder)
            else:
                ax.ax.plot(x, y, color = self.lc, lw = self.lw, solid_capstyle="round", label=label, rasterized=rasterized, zorder=zorder)

        if label is not None:
            leg = ax.ax.legend(frameon=False, prop=ax.fig.font, loc='upper left', bbox_to_anchor=(ax.legend_xpos/ax.fig.fig_width_cm, ax.legend_ypos/ax.fig.fig_height_cm), bbox_transform=ax.fig.fig.transFigure)

            for text in leg.get_texts():

                pl.setp(text, color=ax.fig.textcolor)


class Scatter():
    def __init__(self, ax, x=None, y=None, yerr=None, xerr=None, lc=None, lw=None, dashes=None, pt=None, pc=None,
                 ps=None, label=None, errorbar_area=None, rasterized=False, zorder=0):
        self.lc = lc
        self.lw = lw
        self.ps = ps
        self.dashes = dashes
        self.pt = pt
        self.pc = pc
        self.errorbar_area = errorbar_area

        # rasterization does not work with cairo
        if lc is None:
            self.lc = ax.lc

        if lw is None:
            self.lw = ax.lw
        if dashes is None:
            self.dashes = ax.dashes
        if pt is None:
            self.pt = ax.pt

        if ps is None:
            self.ps = ax.ps

        if pc is None:
            self.pc = ax.pc

        if errorbar_area is None:
            self.errorbar_area = ax.errorbar_area

        x = np.array(x)
        y = np.array(y)

        if yerr is not None:
            yerr = np.array(yerr)
            ax.ax.errorbar(x, y, yerr=yerr, elinewidth=1, ecolor=self.lc, fmt='none', capsize=2, mew=1,
                           solid_capstyle='round', solid_joinstyle='round', zorder=zorder)


        if xerr is not None:
            xerr = np.array(xerr)
            ax.ax.errorbar(x, y, xerr=yerr, elinewidth=1, ecolor=self.lc, fmt='none', capsize=2, mew=1,
                           solid_capstyle='round', solid_joinstyle='round', zorder=zorder)



        ax.ax.scatter(x, y, edgecolors=self.lc, marker=self.pt, s=self.ps,
                      linewidths=self.lw, color=self.pc,
                      label=label, rasterized=rasterized, zorder=zorder)

        if label is not None:
            leg = ax.ax.legend(frameon=False, prop=ax.fig.font, loc='upper left', bbox_to_anchor=(
            ax.legend_xpos / ax.fig.fig_width_cm, ax.legend_ypos / ax.fig.fig_height_cm),
                               bbox_transform=ax.fig.fig.transFigure)

            for text in leg.get_texts():
                pl.setp(text, color=ax.fig.textcolor)



class Bar():
    def __init__(self, ax, x, y, yerr=None, ls = 'solid', alpha=0.5, lc=None, lw=None, bl=None):
        try: 
            # if this is ok, then it is a number
            float(x)
            
            x = np.array([x])
            y = np.array([y])
            if yerr is not None:
                yerr = np.array([yerr])
        except:
            x = np.array(x)
            y = np.array(y)
            if yerr is not None:
                yerr = np.array(yerr)
        
            
        self.lc = lc
        self.lw = lw
        
        if lc is None:
            self.lc = ax.lc
        
        if lw is None:
            self.lw = ax.lw
        
        test = matplotlib.colors.ColorConverter()
        color_rgb = np.array(test.to_rgb(self.lc))
        
        #alpha = 0.6
        lc_alpha_blended = color_rgb*alpha + (1-alpha)
    
        ax.ax.bar(x, y, edgecolor=self.lc, lw = self.lw, ls = ls, facecolor=lc_alpha_blended, align='center')

        if yerr is not None:
            ax.ax.errorbar(x, y, yerr=yerr, elinewidth=self.lw,ecolor=self.lc,fmt='none',capsize=self.lw*1.5, mew=self.lw, solid_capstyle='round', solid_joinstyle='round')

        if bl is not None:

            if not np.isnan(y[0]) and not np.isnan(x[0]):

                x_ = ax.xpos/float(ax.fig.fig_width_cm)+ax.plot_width*((x[0]-ax.xmin)/float(ax.xmax-ax.xmin))/ax.fig.fig_width_cm
                y_ = ax.ypos/float(ax.fig.fig_height_cm)+ax.plot_height*((y[0]+np.sign(y[0] + yerr[0])*yerr[0]-ax.ymin)/float(ax.ymax-ax.ymin))/float(ax.fig.fig_height_cm) + np.sign(y[0] + yerr[0])*0.2/float(ax.fig.fig_height_cm)

                pl.figtext(x_, y_, bl, ha = 'center', va = 'center', fontproperties=ax.fig.font, color=ax.fig.textcolor)
           
class hBar():
    def __init__(self, ax, x, y, xerr=None, ls = 'solid', lc=None, lw=None, bl=None):
        try:
            # if this is ok, then it is a number
            float(x)

            x = np.array([x])
            y = np.array([y])
            if xerr is not None:
                xerr = np.array([xerr])
        except:
            x = np.array(x)
            y = np.array(y)
            if xerr is not None:
                xerr = np.array(xerr)


        self.lc = lc
        self.lw = lw

        if lc is None:
            self.lc = ax.lc

        if lw is None:
            self.lw = ax.lw

        test = matplotlib.colors.ColorConverter()
        color_rgb = np.array(test.to_rgb(self.lc))

        alpha = 0.3
        lc_alpha_blended = color_rgb*alpha + (1-alpha)

        ax.ax.barh(y, x, edgecolor=self.lc, lw = self.lw, ls = ls, facecolor=lc_alpha_blended, align='center')

        if xerr is not None:
            ax.ax.errorbar(x, y, xerr=xerr, elinewidth=self.lw,ecolor=self.lc,fmt='none',capsize=self.lw*2, mew=self.lw, solid_capstyle='round', solid_joinstyle='round')

        # if bl is not None:
        #
        #     if not np.isnan(y[0]) and not np.isnan(x[0]):
        #         x_ = ax.xpos/ax.fig.fig_width_cm+ax.plot_width*((x[0]-ax.xmin)/(ax.xmax-ax.xmin))/ax.fig.fig_width_cm
        #         y_ = ax.ypos/ax.fig.fig_height_cm+ax.plot_height*((y[0]+np.sign(y[0] + yerr)*yerr-ax.ymin)/(ax.ymax-ax.ymin))/ax.fig.fig_height_cm + np.sign(y[0] + yerr)*0.2/ax.fig.fig_height_cm
        #
        #         pl.figtext(x_, y_, bl, ha = 'center', va = 'center', fontproperties=ax.fig.font, color=ax.fig.textcolor)


class Mat():
        def __init__(self, ax, mat, extent, origin = 'lower', interpolation='bilinear', colormap = None, zmin=None, zmax=None):

            if colormap == None:
                colormap = ax.colormap

            if zmin == None:
                zmin = ax.zmin

            if zmax == None:
                zmax = ax.zmax

            ax.ax.imshow(np.array(mat), extent=extent, interpolation=interpolation, origin=origin, aspect='auto', cmap=pl.get_cmap(colormap), vmin=zmin, vmax=zmax)

class Surface():
        def __init__(self, ax, mat, extent, interpolation='bilinear'):
            ax.ax.imshow(np.array(mat), extent=extent, interpolation=interpolation, origin='lower', aspect='auto', cmap=pl.get_cmap(ax.colormap), vmin=ax.zmin, vmax=ax.zmax)
