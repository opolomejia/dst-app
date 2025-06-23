import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


directory = "/mount/src/dst-app/src/streamlit/"
#directory = "./"


def plot_training_history(model_files):
    """
    Plots the training history for each model and displays it in Streamlit.
    
    Parameters:
    model_files (dict): A dictionary where the keys are the model names and the values are the file paths.
    """
    data_frames = {model: pd.read_csv(file) for model, file in model_files.items()}
    columns_to_plot = ['accuracy', 'loss', 'val_accuracy', 'val_loss']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        for model, df in data_frames.items():
            if col in df.columns:
                ax.plot(df[col], label=model)
        ax.set_title(f'{col.capitalize()} Comparison')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(col.capitalize())
        ax.legend()
        ax.grid()

    plt.tight_layout()
    st.pyplot(fig)

def text_mining():
    st.subheader("Les modèles de classifications")
    st.markdown(
        """
        <div style="text-align: justify;">
        Pour notre apprentissage supervisé, nous avons un problème de classification multi-classe, 
        car la variable cible (la nature du document) peut prendre 16 labels. Nous prévoyons de tester 
        trois algorithmes couramment utilisés et robustes pour ce type de problème afin de sélectionner
        celui qui performe le mieux : <b>la régression logistique, le SVC (Support Vector Classifier) et 
        le Random Forest.</b>
        <br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Sélection de modèles")
    st.markdown(
        """
        <div style="text-align: justify;">
        La sélection de modèles est un élément central dans le processus de construction de bons modèles 
        de Machine Learning (supervisé) et peut se résumer au choix des meilleurs hyperparamètres.
        Ces hyperparamètres sont des paramètres définis avant l'entraînement du modèle.Pour ce faire, trois 
        techniques principales peuvent être utilisées pour explorer l'espace des hyperparamètres : GridSearchCV,
        RandomSearchCV et BayesSearchCV.
        Les résultats sont présentés ci-dessous et sont très faibles, quel que soit le modèle utilisé : 
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(directory+"text_model_comp_1.png", caption="Résultats de la sélection de modèles", use_container_width=True)


    st.markdown(
        """
        <div style="text-align: justify;">
        Cela indique que le premier pré-traitement des données n'est pas satisfaisant et doit être revu et/ou il 
        faut retraiter les images. Ainsi, un retraitement de l’image (librairie CV2)  a été initié pour essayer 
        d’améliorer l’extraction des textes des images.
        <br><br>
        Les traitements se font sur le chargement d'une image en N&B, le calibrage sur le contraste et la luminosité
        d’une image ainsi que sur la réduction du bruit gaussien d'une image en N&B. Le bruit gaussien apparaît dans
        une image comme des fluctuations aléatoires de l’intensité des pixels. Cela donne à l’image un aspect granuleux, 
        irrégulier ou  neigeux.
        <br><br>
        Face à la lourdeur des traitements, nous avons réduit considérablement l’échantillon et évalué les modèles 
        avec les mêmes hyperparamètres pour un <b>résultat très faible</b>.
        <br><br>
        Face à ce résultat, nous avons écarté le retraitement des images et travaillé sur l’étape de preprocessing
        (la présente note décrit le dernier prétraitement sans les traitements sur les images). Les entraînements 
        et les évaluations se font sur les deux meilleurs modèles à savoir LogisticRegression et RandomForestClassifier 
        avec le dernier prétraitement. Les résultats sont dans le tableau ci-dessous :
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(directory+"text_model_final_comp.png", caption="Résultats de la sélection de modèles après prétraitement", use_container_width=True)
    
    st.subheader("Rapport de classification et matrice de confusion")
    st.markdown(
        """
        <div style="text-align: justify;">
        Dans la suite, nous présentons uniquement le rapport de classification et la matrice de confusion du modèle 
        <b>LogisticRegression</b> avec le dernier prétraitement, car il s'agit du modèle ayant obtenu les meilleurs 
        résultats lors de nos expérimentations.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(directory+"text_model_class_rep.png", caption="Rapport de classification du modèle LogisticRegression", use_container_width=False)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(directory+"text_model_conf_mat.png", caption="Matrice de confusion du modèle LogisticRegression", use_container_width=True)

def computer_vision():
    st.header("Modélisation avec réseaux des neurones convolutifs")
    st.markdown(
        """
        <div style="text-align: justify;">
        L'objectif de cette section est de présenter les résultats de différents modèles de réseaux neuronaux convolutifs 
        entraînés sur des données open source. L'analyse comprend l'évaluation des performances des modèles, la comparaison 
        des résultats et les observations tirées du processus d'entraînement.
        <br><br>
        Pour toutes les exécutions d'entraînement présentées ci-dessous, nous avons utilisé deux callbacks :
        <ul>
            <li>Si la <i>val_loss</i> stagne pendant 3 époques consécutives avec un delta minimum de 0.01, le taux 
            d'apprentissage est réduit d'un facteur de 0.1, en attendant 4 époques avant de réessayer.</li>
            <li>Si la fonction de perte ne varie pas de 1% après 5 époques, l'entraînement est arrêté.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Test d'un réseau neuronal personnalisé basique")
    st.markdown(
        """
        <div style="text-align: justify;">
        Comme point de départ, nous testons d'abord un réseau personnalisé très basique, avec l'architecture suivante :<br><br>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(directory+"cv_custom_model_arch.png", caption="Architecture du modèle personnalisé basique", use_container_width=False)

    st.markdown(
        """
        <div style="text-align: justify;">
        Ce premier modèle fonctionne mal. Ci-dessous, nous présentons l'évolution de la perte et de la précision pour le modèle :
        </div>
        """,
        unsafe_allow_html=True
    )

    model_files = {
    'Basic Model': directory+"models/training_history/basic_log.csv"}
    plot_training_history(model_files)

    st.markdown(
        """
        <div style="text-align: justify;">
        <b>Les premiers résultats suggèrent que nous avons un problème de surapprentissage.</b> La précision pour l'ensemble d'entraînement
        est proche de 1.0 et la perte proche de 0 ; tandis que la précision pour l'ensemble de validation est faible (0.77) et la 
        perte élevée (2.2). Pour améliorer cela, nous essayons d'ajouter une couche de Dropout au modèle. Ci-dessous les résultats.
        </div>
        """,
        unsafe_allow_html=True
    )

    model_files = {
    'Basic Model': directory+"models/training_history/basic_log.csv",
    'Basic Model + Dropout': directory+"models/training_history/basic_ia_log.csv"
    }
    plot_training_history(model_files)

    st.markdown(
        """
        <div style="text-align: justify;">
        L'ajout de la couche de dropout a amélioré les performances du modèle sur l'ensemble de validation comme prévu. 
        Nous obtenons une précision plus élevée et une perte plus faible que pour le modèle original. 
        <b>Nous aboutissons à un modèle ayant une précision de 0.91 pour l'ensemble d'entraînement et de 0.85 pour l'ensemble de validation.</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Test de transfert learning")

    st.markdown(
        """
        <div style="text-align: justify;">
        Nous testons maintenant une approche de Transfert Learning, en essayant d'utiliser des modèles existants 
        (et pré-entraînés) fournis par TensorFlow. <br><br>
        Le Transfert Learning est une technique d'apprentissage automatique où un modèle pré-entraîné, qui a été 
        entraîné sur un grand ensemble de données, est affiné sur un ensemble de données plus petit et spécifique à 
        une tâche. Dans le contexte des réseaux de neurones, cette approche exploite les capacités d'extraction de 
        caractéristiques des architectures profondes entraînées sur de vastes ensembles de données, réduisant ainsi 
        le besoin de grandes quantités de données étiquetées et de ressources computationnelles. Pour notre projet, 
        <b>nous avons testé les architectures suivantes : Inception V3, ResNet50 V2 et MobileNet V2.</b>
        <br><br>
        Pour ces tests, nous avons utilisé trois approches différentes :
        <ol>
            <li>Gel des paramètres pré-entraînés et entraînement uniquement de la couche de prédiction.</li>
            <li>Réentraînement de l'architecture entière à partir de zéro (initialisation aléatoire).</li>
            <li>Réentraînement de l'architecture en utilisant les poids pré-entraînés comme point de départ.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    training_types = [
        "Entraînement uniquement de la couche de prédiction",
        "Entraînement complet (initialisation aléatoire)",
        "Fine-tuning (poids pré-entraînés, gel partiel)"        
    ]

    selected_training = st.selectbox(
        "Choisissez le type d'entraînement à afficher :",
        training_types
    )

    if selected_training == "Entraînement uniquement de la couche de prédiction":
        st.markdown(
            """
            <div style="text-align: justify;">
            Ci-dessous, nous présentons l'évolution de la précision (accuracy) et 
            de la perte (loss) pour chaque architecture durant le processus d'entraînement : <br><br>
            </div>
            """,
            unsafe_allow_html=True
        )
        model_files = {
        'InceptionV3': directory+"models/training_history/freeze_transfert_learn_inception_v3_log.csv",
        'ResNet50V2': directory+"models/training_history/freeze_transfert_learn_resnet50v2_log.csv",
        'MobileNetV2': directory+"models/training_history/freeze_transfert_learn_mobilenetv2_1.00_224_log.csv",
        'Custom': directory+"models/training_history/basic_ia_log.csv"
        }
        plot_training_history(model_files)

        st.markdown(
            """
            <div style="text-align: justify;">
            <b>Aucun des modèles standards ne semble capable de surpasser notre modèle initiale, 
            que ce soit en termes de précision ou de perte</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif selected_training == "Entraînement complet (initialisation aléatoire)":
        st.markdown(
            """
            <div style="text-align: justify;">
            Pour le deuxième test, nous avons utilisé l'architecture standard comme point de départ et entraîné 
            tous les paramètres à partir de zéro (initialisation aléatoire). Cette approche consomme plus de ressources 
            et nécessite plus de temps d'entraînement comparé à l'utilisation de paramètres pré-entraînés.<br>
            Ci-dessous, nous présentons l'évolution de la précision (accuracy) et de la perte (loss) pour chaque 
            architecture durant le processus d'entraînement.
            </div>
            """,
            unsafe_allow_html=True
        )

        model_files = {
        'InceptionV3': directory+"models/training_history/inception_v3_log.csv",
        'ResNet50V2': directory+"models/training_history/resnet_log.csv",
        'MobileNetV2': directory+"models/training_history/mobilenetv2_1.00_300_log.csv",
        'Custom': directory+"models/training_history/basic_ia_log.csv"
        }
        plot_training_history(model_files)

        st.markdown(
            """
            <div style="text-align: justify;">
            Pendant les 10 premières époques, seul le modèle InceptionV3 semble obtenir de meilleures 
            performances que l'architecture de base avec dropout. Cependant, après 10 époques, l'architecture 
            de base est surpassée par toutes les architectures standards. 
            <b>Le modèle InceptionV3 semble être le meilleur en termes de précision et de perte pour l'ensemble de validation.</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif selected_training == "Fine-tuning (poids pré-entraînés, gel partiel)" :
        st.markdown(
            """
            <div style="text-align: justify;">
            Pour ce troisième test, nous entraînons à nouveau les architectures standards, mais cette fois-ci, 
            nous utilisons les valeurs des paramètres pré-entraînés comme point de départ.
            </div>
            """,
            unsafe_allow_html=True
        )

        model_files = {
        'InceptionV3': directory+"models/training_history/transfert_learn_inception_v3_log.csv",
        'ResNet50V2': directory+"models/training_history/transfert_learn_resnet50v2_log.csv",
        'MobileNetV2': directory+"models/training_history/transfert_learn_mobilenetv2_1.00_224_log.csv",
        'Custom': directory+"models/training_history/basic_ia_log.csv"
        }

        plot_training_history(model_files)

        st.markdown(
            """
            <div style="text-align: justify;">
            <b>Une fois de plus, le meilleur modèle semble être InceptionV3, offrant la plus haute précision pour 
            l'ensemble de validation (0.91) et la deuxième meilleure perte (0.44).</b>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader("Rapport de classification et matrice de confusion")

    st.markdown(
        """
        <div style="text-align: justify;">
        Dans la suite, nous présentons uniquement le rapport de classification et la matrice de confusion du modèle 
        <b>InceptionV3</b> avec le fine-tuning, car il s'agit du modèle ayant obtenu les meilleurs résultats lors de nos expérimentations.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(directory+"cv_class_rep.png", caption="Rapport de classification du modèle InceptionV3", use_container_width=False)
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(directory+"cv_conf_matrix.png", caption="Matrice de confusion du modèle InceptionV3", use_container_width=True)

def clip():
    st.subheader("Modeles de classification avec CLIP")
    st.markdown(
        """
        <div style="text-align: justify;">
        Le dernier modèle de classification a été évalué à l’aide de CLIP (Contrastive Language-Image Pre-Training)
        d’OpenAI, un réseau neuronal entraîné sur des paires image-texte. Les résultats obtenus, présentés ci-dessous, 
        ne surpassent guère ceux des modèles présentés avant. Étant donné la précision obtenue, nous avons choisi de ne
        pas poursuivre les tests sur les ensembles de validation. <b>Pour les tests nous avons chargé le modèle “Vit-B/32”</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    clip_options = [
        "Test 1 : Images et textes bruts",
        "Test 2 : Redimensionnement des images",
        "Test 3 : Redimensionnement + contraste",
        "Test 4 : Redimensionnement + contraste + contour"
    ]
    selected_clip_option = st.selectbox("Choisissez le test CLIP à afficher :", clip_options)

    if selected_clip_option == "Test 1 : Images et textes bruts":
        st.subheader("Entraînement sur les images et textes brutes")
        st.markdown(
            """
            <div style="text-align: justify;">
            Nous avons commencé par entraîner le modèle sur les images et les textes bruts, sans aucun prétraitement. 
            Les résultats sont présentés ci-dessous :
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(directory+"clip_distr_classes_brut.png", caption="Distribution des classes réelles et prédites sans prétraitement", 
                use_container_width=True)

        st.markdown(
            """
            <div style="text-align: justify;">
            Le modèle a obtenu une <b>précision de 0.08 </b> sur l'ensemble d'entraînement.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif selected_clip_option == "Test 2 : Redimensionnement des images":
        st.subheader("Entraînement sur les images et textes en redimensionnant")
        st.markdown(
            """
            <div style="text-align: justify;">
            Pour ce second test, nous avons réduit l’échantillon à 15 000 afin d’obtenir un retour rapide, sachant que l'entraînement 
            du modèle précédent a duré 48 heures. De plus, nous avons redimensionné les images en 256 × 256.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(directory+"clip_distr_classes_resize.png", caption="Distribution des classes réelles et prédites avec redimensionnement", 
                use_container_width=True)

        st.markdown(
            """
            <div style="text-align: justify;">
            Le modèle a obtenu une <b>précision de 0.08 </b> sur l'ensemble d'entraînement.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif selected_clip_option == "Test 3 : Redimensionnement + contraste":
        st.subheader("Entraînement sur les images et textes en redimensionnant et avec constraste")

        st.markdown(
            """
            <div style="text-align: justify;">
            Pour ce troisième test, nous avons appliqué un redimensionnement des images en 256 × 256 et un calibrage sur le contraste à 2.0. 
            Les résultats sont présentés ci-dessous :
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(directory+"clip_distr_classes_resize_contrast.png", 
                 caption="Distribution des classes réelles et prédites avec redimensionnement et contraste",
                 use_container_width=True)

        st.markdown(
            """
            <div style="text-align: justify;">
            Le modèle a obtenu une <b>précision de 0.08 </b> sur l'ensemble d'entraînement.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif selected_clip_option == "Test 4 : Redimensionnement + contraste + contour":
        st.subheader("Entraînement sur les images et textes en redimensionnant, avec contraste et contour")

        st.markdown(
            """
            <div style="text-align: justify;">
            Pour ce quatrième test, nous avons appliqué un redimensionnement des images en 256 × 256, un calibrage sur le contraste
            à 1.5 et en ajoutant des filtres sur les contours,  ci-dessous les résultats obtenus:
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(directory+"clip_distr_classes_resize_contrast_contour.png", 
                 caption="Distribution des classes réelles et prédites avec redimensionnement, contraste et contour",
                 use_container_width=True)

        st.markdown(
            """
            <div style="text-align: justify;">
            Le modèle a obtenu une <b>précision de 0.08 </b> sur l'ensemble d'entraînement.
            </div>
            """,
            unsafe_allow_html=True
        )
        
def main():
    st.title("Modelisation")
    # Create tabs
    text_mining_tab, cv_tab, clip_tab = st.tabs(["Text Mining", "Computer Vision", "CLIP"])
    with text_mining_tab:
        text_mining()

    with cv_tab:
        computer_vision()
    
    with clip_tab:
        clip()

    
if __name__ == "__main__":
    main()