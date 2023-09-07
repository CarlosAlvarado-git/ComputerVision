import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def imgview(img, title=None, filename=None, axis=False):
    """
        Variables:
            img, array de la image
            tittle, string del titulo de la imangen
            filename, nombre con el que se guardaría la imagen
            axis
        Retorno:
            printea la imagen que fue enviada con sus parámetros agregados
    """
    r,c = img.shape[0:2]
    k=8
    fig = plt.figure(figsize=(k,k)) #creamos una figura. Es crear una instancia donde vamos a dibujar. 
    ax = fig.add_subplot(111) # sobre la misma, agrego un subplot(111), el 111 es la posición. filas, columnas y posición a la cual nos referimos el ax. 
                              # el ax, (axis) son ejes. 
    if len(img.shape) == 3:  #decido si la imagen es o no a color. Y decimos cómo dibujarla. 
        im = ax.imshow(img,extent=None) # punto el contenido del ax. extent, configurar el origen de la gráfica. 
    else:
        im = ax.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255) # escala de color, min y max de la escala.
    # el im, sirve para reciclar el objeto. 

    if title != None:
        ax.set_title(title,fontsize=14) # si  trae un titulo, se lo pongo a la imagen. 
    
    
    if not axis: # para poner las "direcciones" o la posición de cada pixel de la imagen. 
        plt.axis('off')
    else:
        ax.grid(c='w')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_xlabel('Columns',fontsize=14)
        ax.set_ylabel('Rows',fontsize=14)
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w',labelsize=14)
        ax.tick_params(axis='y', colors='w',labelsize=14)
        
    if filename != None: # si tiene un dirección, entonces lo guarda ahí. 
        plt.savefig(filename)
    plt.show()

def hist(img,title=None,fill=False,axis=False,filename=None):
    """_summary_
        Muestra la distrubución de la escala de la imagen. Asi vemos que como esta distribuida.
    Args:
        img (array int): array de la imagen
        fill (bool, optional): _description_. Defaults to False.
        filename (string, optional): nombre de la imagen al ser guardada. Defaults to None.
    """
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(122)
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        colors = ['r','g','b']
        for i, color in enumerate(colors):
            histr = cv.calcHist([img],[i],None,[256],[0,256])
            ax.plot(histr, c=color, alpha=0.9)
            x = np.arange(0.0, 256, 1)
            if fill:
                ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color=color)
        
    else:
        histr = cv.calcHist([img],[0],None,[256],[0,256])
        ax.plot(histr, c='w', alpha=0.9)
        x = np.arange(0.0, 256, 1)
        if fill:
            ax.fill_between(x, 0, histr.ravel(), alpha=0.2, color='w')


    ax.set_xlim([0,256])
    ax.grid(alpha=0.2)
    ax.set_facecolor('k')
    ax.set_title('Histogram', fontsize=20)
    ax.set_xlabel('Pixel value', fontsize=20)
    ax.set_ylabel('Pixel count', fontsize=20)
    ax2 = fig.add_subplot(121)
    if len(img.shape) == 3:  #decido si la imagen es o no a color. Y decimos cómo dibujarla. 
        im = ax2.imshow(img,extent=None) # punto el contenido del ax. extent, configurar el origen de la gráfica. 
    else:
        im = ax2.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255) # escala de color, min y max de la escala.
    
    # el im, sirve para reciclar el objeto. 

    if title != None:
        ax2.set_title(title,fontsize=14) # si  trae un titulo, se lo pongo a la imagen. 
    
    
    if not axis: # para poner las "direcciones" o la posición de cada pixel de la imagen. 
        plt.axis('off')
    else:
        ax2.grid(c='w')
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top') 
        ax2.set_xlabel('Columns',fontsize=14)
        ax2.set_ylabel('Rows',fontsize=14)
        ax2.xaxis.label.set_color('w')
        ax2.yaxis.label.set_color('w')
        ax2.tick_params(axis='x', colors='w',labelsize=14)
        ax2.tick_params(axis='y', colors='w',labelsize=14)
    

    if filename != None:
        plt.savefig(filename)
    plt.show()

def splitrgb(img, filename=None):
    """_summary_
        funcion que imprime la imagen recibida en: 
            - A color
            - Escala R
            - Escala G
            - Escala B
    Args:
        img (array int): array de la imagen
        filename (string, None): es el nombre de la imagen al ser guardada. Defaults to None.

    Returns:
        hace el plot de la figura creada con las cautro imágenes. 
    """
    if len(img.shape) != 3: # que sea color
        return 0
    
    fig = plt.figure(figsize=(10,10)) # hacer la figura
    # obtener los 3 canales de color que hay. 
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    # a la figura, le hago el subplot, los 2, 2. Son las filas y columnas, y el 1 es "la primera imagen."
    ax1 = fig.add_subplot(221)
    ax1.imshow(img)
    ax1.set_title('RGB')
    plt.axis('off') # adios esquinas y mediciones para la posicion de los pixeles.
    ax2 = fig.add_subplot(222)
    ax2.imshow(r,cmap='gray', vmin=0, vmax=255) # pongo el plano y la escala que quiero. 
    ax2.set_title('R')
    plt.axis('off') # adios esquinas y mediciones para la posicion de los pixeles.
    ax3 = fig.add_subplot(223)
    ax3.imshow(g,cmap='gray', vmin=0, vmax=255) # pongo el plano y la escala que quiero. 
    ax3.set_title('G')
    plt.axis('off') # adios esquinas y mediciones para la posicion de los pixeles.
    ax4 = fig.add_subplot(224)
    ax4.imshow(b,cmap='gray', vmin=0, vmax=255) # pongo el plano y la escala que quiero. 
    ax4.set_title('B')
    plt.axis('off')
    if filename != None: # si tiene un dirección, entonces lo guarda ahí. 
        plt.savefig(filename)
    plt.show()


def imgcmp(img, img2, title=None, axis=False):
    """_summary_
        funcion que recibe dos array que representan las imagenes y las pone en una misma figura.
        valido si son a color o no para definir como mostrarlas.
    Args:
        img (array int): array de la primera imagen
        img2 (array int): array de la segunda imagen
    """
    fig = plt.figure(figsize=(10,10)) # hacer la figura
    ax1 = fig.add_subplot(121)
    if len(img.shape) == 3:  #decido si la imagen es o no a color. Y decimos cómo dibujarla. 
        ax1.imshow(img,extent=None) # punto el contenido del ax. extent, configurar el origen de la gráfica. 
    else:
        ax1.imshow(img,extent=None,cmap='gray',vmin=0,vmax=255) # escala de color, min y max de la escala.
    if title != None:
        if len(title) > 0:
            ax1.set_title(title[0])
        else:
            ax1.set_title("Imagen 1")
    if not axis: # para poner las "direcciones" o la posición de cada pixel de la imagen. 
        plt.axis('off')
    else:
        ax1.grid(c='w')
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top') 
        ax1.set_xlabel('Columns',fontsize=14)
        ax1.set_ylabel('Rows',fontsize=14)
        ax1.xaxis.label.set_color('w')
        ax1.yaxis.label.set_color('w')
        ax1.tick_params(axis='x', colors='w',labelsize=14)
        ax1.tick_params(axis='y', colors='w',labelsize=14)
    ax2 = fig.add_subplot(122)
    if len(img2.shape) == 3:  #decido si la imagen es o no a color. Y decimos cómo dibujarla. 
        ax2.imshow(img2,extent=None) # punto el contenido del ax. extent, configurar el origen de la gráfica. 
    else:
        ax2.imshow(img2,extent=None,cmap='gray',vmin=0,vmax=255) # escala de color, min y max de la escala.
    if title != None:
        if len(title) > 1:
            ax2.set_title(title[1])
        else:
            ax2.set_title("Imagen 2")
    if not axis: # para poner las "direcciones" o la posición de cada pixel de la imagen. 
        plt.axis('off')
    else:
        ax2.grid(c='w')
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top') 
        ax2.set_xlabel('Columns',fontsize=14)
        ax2.set_ylabel('Rows',fontsize=14)
        ax2.xaxis.label.set_color('w')
        ax2.yaxis.label.set_color('w')
        ax2.tick_params(axis='x', colors='w',labelsize=14)
        ax2.tick_params(axis='y', colors='w',labelsize=14)
    plt.show()
def imgcdf(img):
    """Compute the CDF on an image, va caulculando la suma de las veces que encuentra en el histograma 
            un nuevo "color"/tonalidad de la imagen. 
    Args: 
        img (numpy array): Source image
    Returns:
        cdf (list): Computed CDf of img
        hist (list): Histogram of img
    """
    hist_list = cv.calcHist([img],[0],None,[256],[0,256])
    #hist = [(item) for sublist in hist_list for item in sublist]
    hist = hist_list.ravel()

    # define cdf placeholder
    cdf = []
    t = 0
    for p in hist:
        t += p
        cdf.append(t)
    return cdf, hist


def imgeq(img): 
    """ Equializa la imagen, lo que nos permite tener una mejor distinción de las figuras a lo largo de la imagen
    Args:
        img (numpy array): Grayscale image to equalize
    Returns:
        eq (numpy array): Equalized image
    """
    cdf = imgcdf(img)[0]
    cdf_eq = []
    n = img.shape[0] * img.shape[1] #tamaño de la imagen
    m = min(i for i in cdf if i > 0) # min != 0 

    for i in cdf:
        if i >= m:
            cdf_eq.append(int(round(255*(i-m)/(n-m)))) #normalizamos el cdf
        else:
            cdf_eq.append(0)
    eq = cv.LUT(img, np.array(cdf_eq).astype(np.uint8))
    return eq

def cdfview(cdf, hist, filename=None):
    """Plots the CDF and histogram in a shared x axis
    Args:
        cdf [list]: CDF value to compare
        hist [list]: Histogram to compare
        filename [string]: Optional filename to save output
    Returns:
        None
    """
    fig, ax1 =plt.subplots(figsize=(12,8))
    ax2 = ax1.twinx()
    ax1.plot(hist, label='Value Prob',c='w', lw=0.9)
    ax2.plot(cdf, c='g', label='CDF')
    ax1.set_facecolor('k')
    ax1.set_title('Histogram Vs. CDF')
    ax2.legend()
    if filename != None:
        plt.savefig(filename)
    plt.show()
    return None