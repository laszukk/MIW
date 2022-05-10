import math
import numpy as np
import random as rnd
 
 
def wczytaj(source):
    lista = []
    with open(source) as ausF:
        for i in ausF:
            lista.append(list(map(lambda e: float(e), i.replace('\n', '').split())))
    return lista
 
 
def euklides(a, b):
    tmp = 0
    for x in range(len(a)):
        if (x != len(a) - 1):
            tmp = tmp + (a[x] - b[x]) ** 2
    return math.sqrt(tmp)
 
 
lista = wczytaj("../GK/australian.dat")
 
macierz = []
with open("../GK/australian.dat", "r") as file:
    macierz = [list(map(lambda a: float(a), line.split())) for line in file]
 
 
# licznik=0
# for x in wczytaj(("D:/kl2/australian.dat")):
#     print(x)
#     licznik=licznik+1
#     if(licznik==5):
#         break
# print(euklides(lista[0],lista[1]))
# y=lista[0]
# d(x,y), gdzie x nalezy lista, bez elementu z indeksem 0
# pogrupowac klasa indentyfikacyjna x(ostatnia dana) lista odlegosci
 
def wTuple(x, lista):
    listaTupla = []
    statusList = []
    for z in lista:
        statusList.append(z[-1])
    # print(statusList)
    for y in range(len(lista)):
        val = euklides(x, lista[y])
        listaTupla.append((statusList[y], val))
 
    return listaTupla
 
 
def toDict(lista):
    slownik = {}
    for a, b in lista:
        slownik.setdefault(a, []).append(b)
    return slownik
 
 
def altDict(lista):
    slownik = {}
    for para in lista:
        c = para[0]
        if (c not in slownik):
            slownik[c] = []
        slownik[c].append(para[1])
    return slownik
 
 
def min(lista):
    min = lista[0]
    index = 0
    for x in range(len(lista)):
        if min > lista[x]:
            min = lista[x]
            index = x
    return index
 
 
def sum(slownik, k):
    lista = []
    for x in slownik.keys():
        tmp = list(slownik.get(x))
        tmp.sort()
        lista.append(tmp)
    sumy = []
    for x in lista:
        tmp = 0
        for y in range(k):
            tmp = tmp + x[y]
        sumy.append(tmp)
    print(sumy)
    index = min(sumy)
    klucze = list(slownik.keys())
    print("Najblizej do", klucze[index])
 
 
def euklidesskal(a, b):
    a.pop()
    b.pop()
    v1 = np.array(a)
    v2 = np.array(b)
    return math.sqrt(np.dot((v1 - v2), (v1 - v2)))
 
 
tup = wTuple(lista[0], lista)
lista2 = altDict(tup)
sum(lista2, 3)
# print(euklides(lista[0], lista[2]))
# print(euklidesskal(lista[0], lista[2]))
 
 
def funkcja(x):
    return x
 
 
def montecarloCalk(funkcja, a, b, n=1000):
    wartosci = np.random.uniform(a, b, n)
    y = [funkcja(val) for val in wartosci]
    y_srednia = np.sum(y) / n
    calka = (b - a) * y_srednia
 
    return calka
 
 
def numerczyneCalk(funkcja, a, b, i):
    przedzial = (b - a) / i
    calka = 0
    for x in range(i):
        x = x * przedzial + a
        fx1 = eval(funkcja)
        x += przedzial
        fx2 = eval(funkcja)
        calka = calka + 0.5 * przedzial * (fx1 + fx2)
    return calka
 
 
# print(montecarloCalk(funkcja, 0, 1, 500000))
# print(numerczyneCalk("x", 0, 1, 3))
 
 
def srednia_aryt(lista):
    return sum(lista) / len(lista)
 
 
def wariancja(lista):
    srednia = srednia_aryt(lista)
    return sum((xi - srednia) ** 2 for xi in lista) / len(lista)
 
 
def odchylenie(lista):
    return math.sqrt(wariancja(lista))
 
 
macierztmp = [x[:14] for x in macierz]
# print(srednia_aryt(macierztmp[0]))
# print(wariancja(macierztmp[0]))
# print(odchylenie(macierztmp[0]))
 
#policz sumaryczna odleglosc kropki od pozostatalych
#z petla 1. odleglosc miedzy kropkami euklides (waga jak ta suma i srodek ciezkosci to najmniejsza suma)
#2. przemalowywanie (porowananie do dwoch srodkow ciezkosci 3.i znowu srodek ciezkosci i az nie beda przemalowane
#3 arytmetyczna z np.dot
# 1/n * [x1 x2 ... xn] iloczyn skalar [1 1 ... 1]
# wariancja 1/n * [x1 x2 ... xn] iloczyn skalar [1 1 ... 1] -> a [x1-xsrednia x2-srednia ... xn-srednia]
 
 
# B0+B1x1=y1
# B1 kąt nachylenia
# B0 podnosi (punkt gdzie prosta przecina)
# element neutralny a+e=e+a=a
# (XTX-1) | XTX*B=XTy (XTX)-1 - XT- transpozycja
# B=(XTX)-1*XTy
# 2/7 5/14

def marker(macierz,k):
    for wiersz in macierz:
        for x in range(15):
            if x==14:
                wiersz[x]=float(rnd.randrange(0,k))
    return macierz

def centroidy(lista,macierz,k):
    num=[]
    for x in range(len(lista)):
        num.append([lista[x],x])
    sor=sorted(num, key=lambda x: x[0])
    hlp=[]
    licznik=0
    for z in range(k-1):
        for wiersz in sor:
            if(float(licznik)==macierz[(wiersz[1])][14]):
                hlp.append(wiersz)
                licznik=licznik+1
    return hlp            



def podzial(macierz):
    k=2
    pomalowane=marker(macierz,k)
    for x in range(4):
        wagi=[]
        for wiersz in pomalowane:
            suma=0
            for wiersz2 in pomalowane:
                if wiersz[14]==wiersz2[14]:
                    odl=euklides(wiersz,wiersz2)
                    suma=suma+odl
            wagi.append(suma) 
        srodki=centroidy(wagi,macierz,k)
        for wiersz3 in pomalowane:
            tmp=[]
            for wiersz4 in srodki:
                odl=euklides(wiersz3,pomalowane[wiersz4[1]])
                tmp.append(odl)
            indeks=min(tmp)
            if(indeks!=wiersz3[14]):
                wiersz3[14]==float(indeks)   
    return pomalowane        

#print(macierz)
#print(marker(macierz,2))
print(podzial(macierz))

def m_kowariancji(matrix):
    return np.dot(matrix.T,matrix)

def odwrotnosc_m(matrix):
    return np.linalg.inv(matrix)

def lewa_odwrotnosc(matrix):
    kow = m_kowariancji(matrix)
    odwrotnosc = odwrotnosc_m(kow)
    return np.dot(odwrotnosc,matrix.T)
  
def regresja_liniowa(matrix):
    matrix_x=np.array([[1,x[0]]for x in matrix])
    matrix_y=np.array([x[1]for x in matrix])
    lewa_odw = lewa_odwrotnosc(matrix_x)
    return np.dot(lewa_odw,matrix_y)
    

x_y=np.array([[2,1],[5,2],[7,3],[8,3]])
print(regresja_liniowa(x_y))
 
def projection(u,v):
    uTv = np.dot(u.T,v)
    uTu = np.dot(u.T,u)
    return (uTv/uTu)*u

def matrix_len(u):
    return math.sqrt(np.dot(u.T,u))

a=np.array([[0,1],[0,0],[1,1]])

def qr(a):
    vlista=[ [ x[i] for x in a ] for i in range(len(a[1])) ]
    ulista = []
    q = []

    for x in vlista:
        x = np.array(x)
        projsuma = 0
        for u in ulista:
            utmp = np.array(utmp)
            projsuma=projsuma+projection(utmp, x)
        u = x - projsuma
        ulista.append(u)
        e = (1/matrix_len(u))*u
        q.append(e)
     
    q = np.array(q).T
    r = np.dot(q.T,a)

    return r,q
        
print(qr(a))
 