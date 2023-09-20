import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cvlib
import statistics as st
import argparse



def view(img, title=None):
    k = 5
    fig,ax1 = plt.subplots(figsize=(k,k))
    if len(img.shape)==2:
        ax1.imshow(img, vmin=0, vmax=255, cmap='gray')
    else:
        ax1.imshow(img)
    if title:
        plt.title(title)   
    plt.axis('off')
    plt.show()



# path = '/Users/carlosalvarado/Desktop/ComputerVision/proyecto1/fotos_placas/'
# filename = path+'placa9.jpg'
# im = cv.imread(filename)
def main():
    image_path = ''
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--p", help="image path")

    args = argParser.parse_args()
    print("args=%s" % args)
    print("args.p=%s" % args.p)
    image_path = args.p
    print ('image_path is ', image_path)
    #path = '/Users/carlosalvarado/Desktop/ComputerVision/proyecto1/fotos_placas/'
    filename = image_path
    print(filename)
    # si uso
    plt.style.use('dark_background') 
    im = cv.imread(filename)
    assert im is not None, "file could not be read, check with os.path.exists()"
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    print(imgray.shape)
    h_img = imgray.shape[0]
    w_img = imgray.shape[1]
    # # floddfill
    # im_floodfill = imgray.copy()
    # D = 65
    # cv.floodFill(im_floodfill, None, (0 ,0), 255,  loDiff=D, upDiff=D, flags=cv.FLOODFILL_FIXED_RANGE)
    b = 5
    #blur 
    img_blur = cv.blur(imgray,(b,b))
    # adaptive Th
    imgbin = cv.adaptiveThreshold(img_blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,5)

    # compute contours
    mode = cv.RETR_TREE # contour retrieval mode
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] # contour approximation method 
    contours, hierarchy = cv.findContours(imgbin, mode, method[1])

    # detecto si tengo placa
    contorno = []
    for c in contours:
        # obtner extent
        area = cv.contourArea(c)
        x,y,w,h = cv.boundingRect(c)
        rect_area = w*h
        extent = float(area)/rect_area
        if (extent > 0.6): #pongo el mínimo del área
            if (w > (h*1.35) or w >= (h*1.2)) and ((w > 100 and w < 600) and (h > 50 and h < 330)):
                print(f"el h: {h}")
                print(f"el w: {w}")
                contorno.append(c)

    contorno_placa = []
    count = 0
    centro = [h_img/2, w_img/2]
    print(centro)
    for c in contorno:
        if (count == 0):
            contorno_placa.append(c)
            count = 1
        else:
            x,y,w,h = cv.boundingRect(contorno_placa[0])
            x1,y1,w1,h1 = cv.boundingRect(c)
            xr1 = centro[1] - x
            xr2 = centro[1] - x1
            if (xr1 < xr2):
                pass
            else:
                contorno_placa.pop()
                contorno_placa.append(c)
    print(len(contorno_placa))

    # pongo los contornos con color
    colores = {}
    indexes = []
    thickness = 1
    #(255,0,0),602:(0,255,0)
    for cnt in range(len(contorno_placa)):
        #print(f"{w}, {h}")
        colores.update({cnt:(0,255,0)})
        indexes.append(cnt)


    # valido si hay placa
    if (len(contorno_placa) != 0): # si tengo placa, extraigo
        placa = im.copy()
        x,y,w,h = cv.boundingRect(contorno_placa[0])
        placa = placa[y:y+h, x:x+w]
        #view(placa)
        imgray = cv.cvtColor(placa, cv.COLOR_BGR2GRAY)
        print(imgray.shape)
        h_img = imgray.shape[0]
        w_img = imgray.shape[1]
        centro = [int(h_img/2), int(w_img/2)]
        print(centro[0])
        print(centro[1])
        # floddfill
        im_floodfill = imgray.copy()
        D = 65
        cv.floodFill(im_floodfill, None, (centro[1] ,centro[0]), 255,  loDiff=D, upDiff=D, flags=cv.FLOODFILL_FIXED_RANGE)
        # b = 5
        # #blur 
        # img_blur = cv.blur(imgray,(b,b))
        # adaptive Th
        imgbin = cv.adaptiveThreshold(im_floodfill, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,51,5)

        # compute contours
        mode = cv.RETR_TREE # contour retrieval mode
        method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE] # contour approximation method 
        contours, hierarchy = cv.findContours(imgbin, mode, method[1])

        contorno = []
        areas = []

        for c in contours:
            # limitar el contorno, para tener un área más ajustada. 
            percent =0.052
            epsilon = percent*cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,epsilon,True)
            if cv.contourArea(approx) > 50:
                areas.append(cv.contourArea(approx))

        moda = st.mode(areas)
        print(areas)
        print(moda)
        for c in contours:   
            # limitar el contorno, para tener un área más ajustada. 
            percent =0.052
            epsilon = percent*cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,epsilon,True)
            if (cv.contourArea(approx)>130): #pongo el mínimo del área
                area = cv.contourArea(c)
                x,y,w,h = cv.boundingRect(c)
                if (w > (h*1.35) or w >= (h*1.2)):
                    pass
                else:
                    if (h > (w*1.35) or h >= (w*1.2)):
                        contorno.append(c)

        colores = {}
        indexes = []
        thickness = 1
        #(255,0,0),602:(0,255,0)
        for cnt in range(len(contorno)):
            #print(f"{w}, {h}")
            colores.update({cnt:(0,255,0)})
            indexes.append(cnt)

        r = placa.copy()
        # cv.rectangle(im.copy(),(x,y),(x+w,y+h),(0,255,0),1)
        for c in indexes:
            x,y,w,h = cv.boundingRect(contorno[c])
            r = cv.rectangle(r,(x,y),(x+w,y+h),colores[c],2)
        view(r)
    else: # si no detecto placa, de una voy a las letras
        contorno = []
        areas = []

        for c in contours:
            # limitar el contorno, para tener un área más ajustada. 
            percent =0.052
            epsilon = percent*cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,epsilon,True)
            if cv.contourArea(approx) > 50:
                areas.append(cv.contourArea(approx))

        moda = st.mode(areas)
        print(areas)
        print(moda)
        for c in contours:   
            # limitar el contorno, para tener un área más ajustada. 
            percent =0.052
            epsilon = percent*cv.arcLength(c,True)
            approx = cv.approxPolyDP(c,epsilon,True)
            if (cv.contourArea(approx)>130): #pongo el mínimo del área
                area = cv.contourArea(c)
                x,y,w,h = cv.boundingRect(c)
                if (w > (h*1.35) or w >= (h*1.2)):
                    pass
                else:
                    if (h > (w*1.35) or h >= (w*1.2)):
                        contorno.append(c)

        colores = {}
        indexes = []
        thickness = 1
        #(255,0,0),602:(0,255,0)
        for cnt in range(len(contorno)):
            #print(f"{w}, {h}")
            colores.update({cnt:(0,255,0)})
            indexes.append(cnt)

        r = im.copy()
        # cv.rectangle(im.copy(),(x,y),(x+w,y+h),(0,255,0),1)
        for c in indexes:
            x,y,w,h = cv.boundingRect(contorno[c])
            r = cv.rectangle(r,(x,y),(x+w,y+h),colores[c],2)
        view(r)


if __name__ == "__main__":
    main()