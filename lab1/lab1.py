#imie = input("Jak masz na imie? ")
#print(f"Czesc, {imie}")

stryng = "1"
calka = 1
zmienno = 1.0

print(type(stryng))
print(type(calka))
print(type(zmienno))

fajne_slowa = ["a", "b", "c", "d"]
print("-".join(fajne_slowa))

print("Ala ma kota i psa".split())


lies = "Metody Inżynierii Wiedzy są najlepsze!!!!!1"
print(lies)
print(len(lies))
print(lies.lower())
print(lies.upper())

unpolishowane = lies.replace("ż", "z").replace("ą", "a")
print(unpolishowane)
print(len(unpolishowane))

unikalne = set(lies)
print(unikalne)
print(len(unikalne))

lista_litery = ["a", "b", "c"]
lista_cyfery = [1, 2, 3]
print(lista_litery + lista_cyfery)

print(lista_litery.index("b"))

lista_litery.append("d")
print(lista_litery)

# macierz odwrotna i kiedy mozna policzyc
