import numpy as np
import collections
import os
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import re

vocabulary_size = 5000  # Maximum szavak száma (ami nem fér bele, egy közös "szemeteskosárra" képezzük: UNK

# órán vett függvény
def build_dataset(words):
    count = [['UNK',
              -1]]  # Countban a szavak és hogy hány van belőlük. Az UNK is egy szó, amelyre az összes ismeretlent képezzük le, értékét később állítjuk be.
    count.extend(collections.Counter(words).most_common(
        vocabulary_size - 1))  # Gyakorisági sorrendbe rendezzük, mert csak a leggyakoribb szavakat tartjuk meg
    dictionary = dict()  # Ez lesz a szó -> egész azonosító leképezés tábla
    for word, _ in count:  # Végigmegyünk az összes szón...
        dictionary[word] = len(dictionary)  # Bekerül a szótárba
    data = list()  # data fogja tartalmazni a szavak számokra leképzett alakját listaként
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK'] a 0. elem
            unk_count += 1  # Számoljuk, hány olyan szó van, amelyet UNK-ra képztünk le (mert ritka)
        data.append(index)
    count[0][1] = unk_count  # UNK 0. elem a countban, annak második tagja az előfordulások száma
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # Visszaadjuk az adatokat (lista, szavak egész azonsítóval benne); a szavak és előfordulásaik számának tömbjét (count),
    # a szótárat (szavak -> szám) és a fordítottját (szám -> szó)
    return data, count, dictionary, reverse_dictionary


data_index = 0

nb_train_per_example = 150
nb_test_per_example = 5
nb_validation_per_example = 25

X_train = []
Y_train = []

X_test = []
Y_test = []

X_validation = []
Y_validation = []

print()

data = os.listdir('Data')
regex = re.compile('[^a-zA-Z]')

# végig megyünk minden fileon

for file in data:
    with open('Data/' + file) as text_file:
        # text_file = open("Data/dickens_1.txt", "r")
        lines = text_file.read().split(' ')
        normalized = []

        # kiszűrünk minden nem betű karaktert
        for word in lines:
            ret = regex.sub('', word)
            if ret != '':
                normalized.append(ret)

        # órán vett módszerrel csinálunk belőle egy szótár tömböt
        data, count, dictionary, reverse_dictionary = build_dataset(normalized)

        # az első mindig a tanító, második a test és harmadik a validációs halmaz
        mode = int(file[2])

        # három különböző halmazba rakjuk őket
        # minták számát feljebb határoztuk meg
        if mode == 1:
            for i in range(nb_train_per_example):
                start = np.random.randint(len(data) - 500)
                output = np.zeros(3)
                output[int(file[0]) - 1] = 1

                X_train.append(data[start: start + 500])
                Y_train.append(output)

        if mode == 2:
            for i in range(nb_test_per_example):
                start = np.random.randint(len(data) - 500)
                output = np.zeros(3)
                output[int(file[0]) - 1] = 1

                X_test.append(data[start: start + 500])
                Y_test.append(output)

        if mode == 3:
            for i in range(nb_validation_per_example):
                start = np.random.randint(len(data) - 500)
                output = np.zeros(3)
                output[int(file[0]) - 1] = 1

                X_validation.append(data[start: start + 500])
                Y_validation.append(output)

x_train = np.array(X_train)
y_train = np.array(Y_train)


# háló összerakása
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_vecor_length, input_length=500))
model.add(LSTM(100))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# tanítás
model.fit(x_train, y_train, nb_epoch=10, batch_size=64, validation_data=(np.array(X_validation), np.array(Y_validation)))

prediction = model.predict(np.array(X_test))

result = np.zeros(shape=(3*nb_test_per_example, 2))

# egy egyszerű kimutatás a várt és a kapott értékekről
for idx, row in enumerate(prediction):
    result[idx][0] = np.argmax(row)

    # a három szerzőtől sorban 5-5 minta volt
    if 0 <= idx < 6:
        result[idx][1] = 0
    if 6 <= idx < 11:
        result[idx][1] = 1
    if 11 <= idx < 16:
        result[idx][1] = 2


# nem igazán tanul de legalább a mienk
print("Kapott | elvart")
print(result)

