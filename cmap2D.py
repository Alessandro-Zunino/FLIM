import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def cmap2d(intensity, lifetime, params):
    
    sz = intensity.shape
    
    Hp = np.minimum( np.maximum(lifetime, params['minTau']), params['maxTau'])
    
    # the HSV representation
    Hn = ( (Hp - params['minTau'])/(params['maxTau'] - params['minTau']) ) * params['satFactor']
    Sn = np.ones( Hp.shape )
    Vn = (intensity - params['minInt'])/(params['maxInt'] - params['minInt'])
    
    HSV = np.empty( (sz[0], sz[1], 3) )
    
    if params['invertColormap'] == True:
        Hn = params['satFactor'] - Hn
    
    # set to violet color of pixels outside lifetime bounds
    # BG = intensity < ( np.max(intensity * params.bgIntPerc )
    Hn[ np.not_equal(lifetime, Hp) ] = params['outOfBoundsHue']
    
    HSV[:,:,0] = Hn.astype('float64')
    HSV[:,:,1] = Sn.astype('float64')
    HSV[:,:,2] = Vn.astype('float64')
    
    #convert to RGB
    RGB = hsv_to_rgb(HSV)

    return RGB


def showFLIM(intensity, lifetime, bounds_Tau = None, bounds_Int = None, satFactor = 0.657, outOfBoundsHue = 0.8, invertColormap = False):
    '''
    Display together the lifetime and intensity images with a proper colormap.
    Referring to the HSV color model:
    Intensity values are mapped in Value
    Lifetime values are mapped in Hue
    If the lifetime of a pixel is outside the interval [minInt, maxInt],
    that pixel is rendered with Hue outOfBoundsHue (default: violet)

    Input parameters
    intensityIm:        Pixel values are photon counts.     size(intensityIm) = [h, w]
    lifetimeIm:         Pixel values are lifetime values.   size(lifetimeIm) = [h, w]

    minTau:             minimum lifetime value of the colorbar ( min(lifetimeIm(:)) ) 
    maxTau:             maximum lifetime value of the colorbar ( max(lifetimeIm(:)) )
    minInt:             minimum intensity value of the colorbar ( min(intensityIm(:)) ) 
    maxInt:             maximum intensity value of the colorbar ( max(intensityIm(:)) ) 
    invertColormap      (true)
    outOfBoundsHue      Hue to render the out of bounds pixels (0.8)
    figDimensions       ( [0,0, size(intensityIm,1), size(intensityIm,2)] )
    satFactor           The span of the Hue space (0.657)
    
    Last Modified May 2022 by Alessandro Zunino
    Based on the MATLAB function written by Giorgio Tortarolo
    '''
    
    
    if bounds_Tau is None:
        bounds_Tau = {
            'minTau' : np.min(lifetime),
            'maxTau' : np.max(lifetime),
            }
        
    if bounds_Int is None:
        bounds_Int = {
            'minInt' : np.min(intensity),
            'maxInt' : np.max(intensity)
            }
    
    params = bounds_Tau.copy()
    params.update(bounds_Int)
    
    params['invertColormap'] = invertColormap
    params['satFactor'] = satFactor
    params['outOfBoundsHue'] = outOfBoundsHue

    sz = intensity.shape
    N = sz[0]
    
    #Image
    
    RGB = cmap2d(intensity, lifetime, params)

    #Colorbar
    
    LG = np.linspace(params['minTau'], params['maxTau'], num = sz[0])
    LifeTimeGradient = np.tile( LG, (N, 1) )
    
    IG = np.linspace(params['minInt'], params['maxInt'], num = N)
    IntensityGradient = np.transpose( np.tile( IG, (sz[0], 1) ) )
    
    RGB_colormap = cmap2d(IntensityGradient, LifeTimeGradient, params) ;
    RGB_colormap = np.moveaxis(RGB_colormap, 0, 1)
    
    # Show combined image with colorbar
    
    extent = (params['minInt'], params['maxInt'], params['minTau'], params['maxTau'])
    
    fig = plt.figure(figsize = (9,8))
    widths = [0.05, 1]
    heights = [1]
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios = widths, height_ratios = heights)
    
    ax = fig.add_subplot(spec[0, 1])
    
    ax.imshow( RGB )
    ax.axis('off')
    
    ax = fig.add_subplot(spec[0, 0])
    
    ax.imshow( RGB_colormap, origin='lower', aspect='auto', extent = extent)
    ax.set_xticks( [params['minInt'], params['maxInt']] )
    ax.set_xlabel('Counts')
    ax.set_ylabel('Lifetime')
    
    plt.tight_layout()

    return RGB, RGB_colormap
