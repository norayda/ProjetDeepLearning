import tensorflow as tf
import numpy as np
import matplotlib.pyplot as  plt
from sklearn.preprocessing import StandardScaler


# Graphe pour afficher les models
def plot_all_images(all_images):
    # Courbe de notre erreur " loss "
    for img in all_images:
        y_coordonne = img[0].history["loss"]
        x_coordonne = list(range(len(y_coordonne)))
        plt.plot(x_coordonne, y_coordonne, label=img[1])
        plt.legend()
        plt.title("Loss")

    plt.show()

    # Courbe de l'accuracy de ce model
    for img in all_images:
        y_coordonne = img[0].history["valeur de l'accuracy"]
        x_coordonne = list(range(len(y_coordonne)))
        plt.plot(x_coordonne, y_coordonne, label=img[1])
        plt.legend()
        plt.title("Accuracy")

    plt.show()


def testerEntre_Conv_et_Dens(build_hidden_layers, modelX, modelY, testX, testY):

    # Creation de notre model d'entrainement en sequentiel
    model = tf.keras.models.Sequential()
    build_hidden_layers(model)

    #model.add(tf.keras.layers.Flatten())

    #modelX = modelX.reshape(-1, 3072)
    #modelX = modelX.astype(float)
    # Mettre les 10 neuronnes de notre
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.sigmoid))

    # Compiler notre model d'entrainement pour optimiser les erreurs de notre model d'entrainement
    model.compile(
        loss=tf.keras.losses.mse,                       # La valeur de l'erreur de notre entrainement
        optimizer=tf.keras.optimizers.SGD(),            # Optimiser l'erreur comme la descente de gradiente
        metrics=tf.keras.metrics.categorical_accuracy,  # Voir le metrics (combien de fois notre model a eu juste sur l'entrainement)
    )

    model.fit(modelX, modelY, validation_data=(testX, testY), epochs=10)

    # Montrer les parametres du model de notre entrainement
    model.summary()


def model_en_Block_De_Conv(model: tf.keras.models.Sequential):

    print("Notre model avec des couches de neuronnes de type blocks de conv-net")
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D())

    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D())


def model_en_Den(model: tf.keras.models.Sequential):

    print("Notre model avec des couches de neuronnes de type dense de neuronnes")

    #model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))



# Fonction principale
if __name__ == '__main__':

    # Recuperation de notre DataSet d'entrainement dans des tableaux suivant :
    # image_inPut  --> Tableau de nos images d'entrainement
    # image_ettiqt --> Tableau des ettiquettes de nos images [0,1,2,3,4,5,6,7,8,9]
    (image_inPut, image_ettiqt), (test_image_inPut, test_image_ettiqt) = tf.keras.datasets.cifar10.load_data()


    # Transformer les ettiquettes des images en un vecteur de valeur 0 ou 1 ----> [0, 0, 0, 1, 0, ..]
    image_ettiqt = tf.keras.utils.to_categorical(image_ettiqt, 10)
    test_image_ettiqt = tf.keras.utils.to_categorical(test_image_ettiqt, 10)

    #model.add(tf.keras.layers.Flatten(input_shape=[32, 32]))

    # Reshappe nos images en unidimentionnel de 32*32*3 = 3072
    # Remplace le flatten de tensorflow

    print("La moyenne des donnees de nos images avant la normalisation est de :", image_inPut.mean())
    print("L'equart type est de : ", image_inPut.std())
    print(" ")

    # Normaliser nos donnees (reduire les pixelles de nos images pour etre proche de 0)
    # Normaliser ses donnees est une bonne pratique a faire surtout pour traiter des donnees qui sont lineaires en entree

    image_inPut = image_inPut.reshape(-1, 3072)
    image_inPut = image_inPut.astype(float)

    test_image_inPut = test_image_inPut.reshape(-1, 3072)
    test_image_inPut = test_image_inPut.astype(float)

    scaler = StandardScaler()
    image_inPut = scaler.fit_transform(image_inPut)
    test_image_inPut = scaler.fit_transform(test_image_inPut)

    print("La moyenne des donnees de nos images apres la normalisation est de :", image_inPut.mean())
    print("L'equart type est de : ", image_inPut.std())

    # Appel de nos fonctions des deux models: Conv-Net & Dense

    testerEntre_Conv_et_Dens(model_en_Den, image_inPut, image_ettiqt, test_image_inPut, test_image_ettiqt)
    #testerEntre_Conv_et_Dens(model_en_Block_De_Conv, image_inPut, image_ettiqt, test_image_inPut, test_image_ettiqt)









