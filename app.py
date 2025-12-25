{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPgEc3WC8v2V4XWuWquerYI",
      
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/HANSVALENTINUS14487/BENGKOD-14487/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yfPk28KIFzxl",
        "outputId": "2fdc0163-667b-428c-def1-96602fe6cd83"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.52.2-py3-none-any.whl.metadata (9.8 kB)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.2.4)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.3.1)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (25.0)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.29.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (9.1.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.45)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.5.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.25.1)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.13.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2025.11.12)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (25.4.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.30.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.52.2-py3-none-any.whl (9.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.0/9.0 MB\u001b[0m \u001b[31m19.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m96.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.52.2\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import joblib\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Config Halaman"
      ],
      "metadata": {
        "id": "e00e1dbUGhRa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.set_page_config(\n",
        "    page_title=\"Prediksi Churn Pelanggan Telco\",\n",
        "    layout=\"wide\"\n",
        ")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cHb_Ug9aGk8U",
        "outputId": "a4a764e6-2efc-46f2-9cfd-346fabb58182"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:43:27.683 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d0212093",
        "outputId": "8b03407b-e8a5-4a25-f28b-194643cd9cf3"
      },
      "source": [
        "%%writefile app.py\n",
        "\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import joblib\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "st.set_page_config(\n",
        "    page_title=\"Prediksi Churn Pelanggan Telco\",\n",
        "    layout=\"wide\"\n",
        ")\n",
        "\n",
        "try:\n",
        "    model = joblib.load(\"best_model.pkl\")\n",
        "except FileNotFoundError:\n",
        "    st.warning(\"Model file 'best_model.pkl' not found. Using a dummy model for demonstration.\")\n",
        "    class DummyModel:\n",
        "        def __init__(self):\n",
        "            self.feature_names_in_ = [\n",
        "                \"SeniorCitizen\", \"tenure\", \"MonthlyCharges\", \"TotalCharges\",\n",
        "                \"gender_Male\", \"Partner_Yes\", \"Dependents_Yes\", \"PhoneService_Yes\",\n",
        "                \"MultipleLines_No phone service\", \"MultipleLines_Yes\",\n",
        "                \"InternetService_DSL\", \"InternetService_Fiber optic\", \"InternetService_No\",\n",
        "                \"Contract_Month-to-month\", \"Contract_One year\", \"Contract_Two year\",\n",
        "                \"PaperlessBilling_Yes\",\n",
        "                \"PaymentMethod_Bank transfer (automatic)\", \"PaymentMethod_Credit card (automatic)\",\n",
        "                \"PaymentMethod_Electronic check\", \"PaymentMethod_Mailed check\"\n",
        "            ]\n",
        "            self.feature_importances_ = np.random.rand(len(self.feature_names_in_))\n",
        "            self.feature_importances_ /= self.feature_importances_.sum() # Normalize to sum to 1\n",
        "\n",
        "        def predict(self, X):\n",
        "            return np.array([0] * len(X))\n",
        "\n",
        "        def predict_proba(self, X):\n",
        "            return np.array([[0.8, 0.2]] * len(X))\n",
        "\n",
        "    model = DummyModel()\n",
        "\n",
        "st.sidebar.title(\"📌 Informasi Aplikasi\")\n",
        "st.sidebar.write(\"\"\"\n",
        "Aplikasi ini digunakan untuk *memprediksi churn pelanggan*\n",
        "pada industri telekomunikasi menggunakan *Machine Learning*.\n",
        "\n",
        "Dataset: Telco Customer Churn\n",
        "Model: Model terbaik hasil Hyperparameter Tuning\n",
        "\"\"\")\n",
        "\n",
        "st.title(\"📊 Prediksi Churn Pelanggan Telco\")\n",
        "st.markdown(\"\"\"\n",
        "Aplikasi ini merupakan hasil *Proyek UAS Data Science*\n",
        "untuk memprediksi apakah seorang pelanggan *berpotensi churn* atau *tidak*.\n",
        "\"\"\")\n",
        "\n",
        "st.subheader(\"🧾 Input Data Pelanggan\")\n",
        "\n",
        "col1, col2, col3 = st.columns(3)\n",
        "\n",
        "with col1:\n",
        "    gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
        "    senior = st.selectbox(\"Senior Citizen\", [0, 1])\n",
        "    partner = st.selectbox(\"Partner\", [\"Yes\", \"No\"])\n",
        "    dependents = st.selectbox(\"Dependents\", [\"Yes\", \"No\"])\n",
        "    tenure = st.number_input(\"Tenure (bulan)\", 0, 72, 12)\n",
        "\n",
        "with col2:\n",
        "    phone = st.selectbox(\"Phone Service\", [\"Yes\", \"No\"])\n",
        "    multiple = st.selectbox(\"Multiple Lines\", [\"Yes\", \"No\", \"No phone service\"])\n",
        "    internet = st.selectbox(\"Internet Service\", [\"DSL\", \"Fiber optic\", \"No\"])\n",
        "    contract = st.selectbox(\"Contract\", [\"Month-to-month\", \"One year\", \"Two year\"])\n",
        "    paperless = st.selectbox(\"Paperless Billing\", [\"Yes\", \"No\"])\n",
        "\n",
        "with col3:\n",
        "    payment = st.selectbox(\n",
        "        \"Payment Method\",\n",
        "        [\n",
        "            \"Electronic check\",\n",
        "            \"Mailed check\",\n",
        "            \"Bank transfer (automatic)\",\n",
        "            \"Credit card (automatic)\"\n",
        "        ]\n",
        "    )\n",
        "    monthly = st.number_input(\"Monthly Charges\", 0.0, 200.0, 70.0)\n",
        "    total = st.number_input(\"Total Charges\", 0.0, 10000.0, 2000.0)\n",
        "\n",
        "input_data = {\n",
        "    \"gender\": gender,\n",
        "    \"SeniorCitizen\": senior,\n",
        "    \"Partner\": partner,\n",
        "    \"Dependents\": dependents,\n",
        "    \"tenure\": tenure,\n",
        "    \"PhoneService\": phone,\n",
        "    \"MultipleLines\": multiple,\n",
        "    \"InternetService\": internet,\n",
        "    \"Contract\": contract,\n",
        "    \"PaperlessBilling\": paperless,\n",
        "    \"PaymentMethod\": payment,\n",
        "    \"MonthlyCharges\": monthly,\n",
        "    \"TotalCharges\": total\n",
        "}\n",
        "\n",
        "input_df = pd.DataFrame([input_data])\n",
        "input_df = pd.get_dummies(input_df)\n",
        "\n",
        "model_features = model.feature_names_in_\n",
        "\n",
        "for col in model_features:\n",
        "    if col not in input_df.columns:\n",
        "        input_df[col] = 0\n",
        "\n",
        "input_df = input_df[model_features]\n",
        "\n",
        "st.subheader(\"🔮 Hasil Prediksi\")\n",
        "\n",
        "if st.button(\"Prediksi Churn\"):\n",
        "    prediction = model.predict(input_df)[0]\n",
        "    probability = model.predict_proba(input_df)[0][1]\n",
        "\n",
        "    if prediction == 1:\n",
        "        st.error(\"⚠ Pelanggan *DIPREDIKSI CHURN*\")\n",
        "    else:\n",
        "        st.success(\"✅ Pelanggan *TIDAK CHURN*\")\n",
        "\n",
        "    st.markdown(f\"*Probabilitas Churn:* {probability:.2%}\")\n",
        "    st.progress(int(probability * 100))\n",
        "\n",
        "st.subheader(\"📊 Feature Importance\")\n",
        "\n",
        "if hasattr(model, \"feature_importances_\"):\n",
        "    importance_df = pd.DataFrame({\n",
        "        \"Fitur\": model.feature_names_in_,\n",
        "        \"Importance\": model.feature_importances_\n",
        "    }).sort_values(by=\"Importance\", ascending=False).head(10)\n",
        "\n",
        "    fig, ax = plt.subplots()\n",
        "    ax.barh(importance_df[\"Fitur\"], importance_df[\"Importance\"])\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"10 Fitur Paling Berpengaruh terhadap Churn\")\n",
        "    ax.set_xlabel(\"Nilai Importance\")\n",
        "\n",
        "    st.pyplot(fig)\n",
        "\n",
        "    st.caption(\"Feature importance diambil dari model Random Forest\")\n",
        "else:\n",
        "    st.info(\"Model tidak mendukung feature importance\")\n",
        "\n",
        "with st.expander(\"📘 Penjelasan Feature Importance\"):\n",
        "    st.write(\"\"\"\n",
        "Feature Importance menunjukkan *fitur-fitur yang paling berpengaruh*\n",
        "dalam menentukan apakah pelanggan akan churn atau tidak.\n",
        "\n",
        "Semakin besar nilai importance, semakin besar pengaruh fitur tersebut\n",
        "terhadap keputusan model.\n",
        "\"\"\")\n",
        "\n",
        "st.markdown(\"---\")\n",
        "st.caption(\"UAS Data Science | Telco Customer Churn Prediction\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4fc1e4ed"
      },
      "source": [
        "!streamlit run app.py &>/dev/null&"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2c8fb765"
      },
      "source": [
        "Klik tautan ngrok di output di atas untuk mengakses aplikasi Streamlit Anda. Jika tautan tidak muncul secara otomatis, Anda mungkin perlu menunggu beberapa saat atau jalankan ulang sel di atas."
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Load Model"
      ],
      "metadata": {
        "id": "NPy79m69HywS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "import numpy as np\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "\n",
        "try:\n",
        "    model = joblib.load(\"best_model.pkl\")\n",
        "except FileNotFoundError:\n",
        "    st.warning(\"Model file 'best_model.pkl' not found. Using a dummy model for demonstration.\")\n",
        "    class DummyModel:\n",
        "        def __init__(self):\n",
        "            self.feature_names_in_ = [\n",
        "                \"SeniorCitizen\", \"tenure\", \"MonthlyCharges\", \"TotalCharges\",\n",
        "                \"gender_Male\", \"Partner_Yes\", \"Dependents_Yes\", \"PhoneService_Yes\",\n",
        "                \"MultipleLines_No phone service\", \"MultipleLines_Yes\",\n",
        "                \"InternetService_DSL\", \"InternetService_Fiber optic\", \"InternetService_No\",\n",
        "                \"Contract_Month-to-month\", \"Contract_One year\", \"Contract_Two year\",\n",
        "                \"PaperlessBilling_Yes\",\n",
        "                \"PaymentMethod_Bank transfer (automatic)\", \"PaymentMethod_Credit card (automatic)\",\n",
        "                \"PaymentMethod_Electronic check\", \"PaymentMethod_Mailed check\"\n",
        "            ]\n",
        "            self.feature_importances_ = np.random.rand(len(self.feature_names_in_))\n",
        "            self.feature_importances_ /= self.feature_importances_.sum() # Normalize to sum to 1\n",
        "\n",
        "        def predict(self, X):\n",
        "            # Dummy prediction: always predict 0 (no churn)\n",
        "            return np.array([0] * len(X))\n",
        "\n",
        "        def predict_proba(self, X):\n",
        "            # Dummy probability: always 80% no churn, 20% churn\n",
        "            # Ensure it returns a 2D array of shape (n_samples, n_classes)\n",
        "            return np.array([[0.8, 0.2]] * len(X))\n",
        "\n",
        "    model = DummyModel()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tFJ0b0a_H0hT",
        "outputId": "bf496ade-17a4-4d07-9eb3-c22f1fa22c95"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:42:38.971 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:42:38.972 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:42:38.979 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sidebar"
      ],
      "metadata": {
        "id": "Uk8pjni2IPN0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.sidebar.title(\"📌 Informasi Aplikasi\")\n",
        "st.sidebar.write(\"\"\"\n",
        "Aplikasi ini digunakan untuk *memprediksi churn pelanggan*\n",
        "pada industri telekomunikasi menggunakan *Machine Learning*.\n",
        "\n",
        "Dataset: Telco Customer Churn\n",
        "Model: Model terbaik hasil Hyperparameter Tuning\n",
        "\"\"\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uxu6AT3bIRx6",
        "outputId": "2a9cc6ca-65a7-4470-b9c2-dc623ceac0cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:29:46.853 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:29:47.003 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-12-25 15:29:47.004 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:29:47.004 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:29:47.006 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:29:47.009 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:29:47.012 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Judul"
      ],
      "metadata": {
        "id": "vkPji-S4IZfW"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.title(\"📊 Prediksi Churn Pelanggan Telco\")\n",
        "st.markdown(\"\"\"\n",
        "Aplikasi ini merupakan hasil *Proyek UAS Data Science*\n",
        "untuk memprediksi apakah seorang pelanggan *berpotensi churn* atau *tidak*.\n",
        "\"\"\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Sw5rxkV9Ic8k",
        "outputId": "1b908614-a36b-4bfe-aedb-80b485103ba8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:30:24.847 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:30:24.848 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:30:24.849 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:30:24.850 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:30:24.851 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:30:24.852 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Form Input"
      ],
      "metadata": {
        "id": "4ofbj4ySIhgH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.subheader(\"🧾 Input Data Pelanggan\")\n",
        "\n",
        "col1, col2, col3 = st.columns(3)\n",
        "\n",
        "with col1:\n",
        "    gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
        "    senior = st.selectbox(\"Senior Citizen\", [0, 1])\n",
        "    partner = st.selectbox(\"Partner\", [\"Yes\", \"No\"])\n",
        "    dependents = st.selectbox(\"Dependents\", [\"Yes\", \"No\"])\n",
        "    tenure = st.number_input(\"Tenure (bulan)\", 0, 72, 12)\n",
        "\n",
        "with col2:\n",
        "    phone = st.selectbox(\"Phone Service\", [\"Yes\", \"No\"])\n",
        "    multiple = st.selectbox(\"Multiple Lines\", [\"Yes\", \"No\", \"No phone service\"])\n",
        "    internet = st.selectbox(\"Internet Service\", [\"DSL\", \"Fiber optic\", \"No\"])\n",
        "    contract = st.selectbox(\"Contract\", [\"Month-to-month\", \"One year\", \"Two year\"])\n",
        "    paperless = st.selectbox(\"Paperless Billing\", [\"Yes\", \"No\"])\n",
        "\n",
        "with col3:\n",
        "    payment = st.selectbox(\n",
        "        \"Payment Method\",\n",
        "        [\n",
        "            \"Electronic check\",\n",
        "            \"Mailed check\",\n",
        "            \"Bank transfer (automatic)\",\n",
        "            \"Credit card (automatic)\"\n",
        "        ]\n",
        "    )\n",
        "    monthly = st.number_input(\"Monthly Charges\", 0.0, 200.0, 70.0)\n",
        "    total = st.number_input(\"Total Charges\", 0.0, 10000.0, 2000.0)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rYSGpyVbIkwZ",
        "outputId": "25016773-dfc8-4bd9-acab-ee3e2eadb3ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:31:01.093 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.095 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.101 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.105 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.108 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.113 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.118 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.122 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.128 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.129 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.131 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.134 Session state does not function when running a script without `streamlit run`\n",
            "2025-12-25 15:31:01.135 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.138 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.139 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.142 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.144 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.145 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.148 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.150 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.152 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.155 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.158 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.159 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.161 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.163 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.166 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.167 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.169 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.177 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.184 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.187 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.188 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.190 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.195 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.196 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.197 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.199 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.201 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.202 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.206 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.209 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.210 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.213 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.216 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.218 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.221 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.224 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.229 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.230 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.234 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.237 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.242 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.249 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.250 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.254 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.259 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.266 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.269 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.272 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.278 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.282 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.286 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.291 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.306 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.309 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.315 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.320 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.327 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.331 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.336 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.345 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.360 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.368 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.369 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.378 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.384 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.397 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.409 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.412 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.415 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.419 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.429 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.434 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.446 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.447 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.459 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.467 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.469 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.472 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.475 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:31:01.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Preprocessing Input"
      ],
      "metadata": {
        "id": "ogxj141mIrez"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "input_data = {\n",
        "    \"gender\": gender,\n",
        "    \"SeniorCitizen\": senior,\n",
        "    \"Partner\": partner,\n",
        "    \"Dependents\": dependents,\n",
        "    \"tenure\": tenure,\n",
        "    \"PhoneService\": phone,\n",
        "    \"MultipleLines\": multiple,\n",
        "    \"InternetService\": internet,\n",
        "    \"Contract\": contract,\n",
        "    \"PaperlessBilling\": paperless,\n",
        "    \"PaymentMethod\": payment,\n",
        "    \"MonthlyCharges\": monthly,\n",
        "    \"TotalCharges\": total\n",
        "}\n",
        "\n",
        "input_df = pd.DataFrame([input_data])\n",
        "input_df = pd.get_dummies(input_df)\n",
        "\n",
        "# Samakan fitur dengan model\n",
        "model_features = model.feature_names_in_\n",
        "\n",
        "for col in model_features:\n",
        "    if col not in input_df.columns:\n",
        "        input_df[col] = 0\n",
        "\n",
        "input_df = input_df[model_features]\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 211
        },
        "id": "WhYEXSv-IvQi",
        "outputId": "db6a5d43-9f00-4c41-a6de-0dd880e510f7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'model' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-408387637.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m \u001b[0;31m# Samakan fitur dengan model\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 21\u001b[0;31m \u001b[0mmodel_features\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfeature_names_in_\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mcol\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mmodel_features\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Prediksi"
      ],
      "metadata": {
        "id": "6lTNpOe5Jad3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.subheader(\"🔮 Hasil Prediksi\")\n",
        "\n",
        "if st.button(\"Prediksi Churn\"):\n",
        "    prediction = model.predict(input_df)[0]\n",
        "    probability = model.predict_proba(input_df)[0][1]\n",
        "\n",
        "    if prediction == 1:\n",
        "        st.error(\"⚠ Pelanggan *DIPREDIKSI CHURN*\")\n",
        "    else:\n",
        "        st.success(\"✅ Pelanggan *TIDAK CHURN*\")\n",
        "\n",
        "    st.markdown(f\"*Probabilitas Churn:* {probability:.2%}\")\n",
        "    st.progress(int(probability * 100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "22HPvLjMJfj9",
        "outputId": "a3f09154-8240-466d-ea81-bb66518d8ba4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:35:00.579 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.586 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.589 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.590 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.597 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.601 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.604 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.607 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:35:00.613 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Feature Importance"
      ],
      "metadata": {
        "id": "mFGFKjYrJmCm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.subheader(\"📊 Feature Importance\")\n",
        "\n",
        "if hasattr(model, \"feature_importances_\"):\n",
        "    importance_df = pd.DataFrame({\n",
        "        \"Fitur\": model.feature_names_in_,\n",
        "        \"Importance\": model.feature_importances_\n",
        "    }).sort_values(by=\"Importance\", ascending=False).head(10)\n",
        "\n",
        "    fig, ax = plt.subplots()\n",
        "    ax.barh(importance_df[\"Fitur\"], importance_df[\"Importance\"])\n",
        "    ax.invert_yaxis()\n",
        "    ax.set_title(\"10 Fitur Paling Berpengaruh terhadap Churn\")\n",
        "    ax.set_xlabel(\"Nilai Importance\")\n",
        "\n",
        "    st.pyplot(fig)\n",
        "\n",
        "    st.caption(\"Feature importance diambil dari model Random Forest\")\n",
        "else:\n",
        "    st.info(\"Model tidak mendukung feature importance\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        },
        "id": "TyVsrnbpJqM7",
        "outputId": "14006ea7-5978-4114-df37-dd61e0263e41"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:36:05.917 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:36:05.919 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:36:05.920 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'model' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-3917220207.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubheader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"📊 Feature Importance\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"feature_importances_\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m     importance_df = pd.DataFrame({\n\u001b[1;32m      5\u001b[0m         \u001b[0;34m\"Fitur\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfeature_names_in_\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Penjelasan"
      ],
      "metadata": {
        "id": "BiDxcChzJ7Di"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "with st.expander(\"📘 Penjelasan Feature Importance\"):\n",
        "    st.write(\"\"\"\n",
        "Feature Importance menunjukkan *fitur-fitur yang paling berpengaruh*\n",
        "dalam menentukan apakah pelanggan akan churn atau tidak.\n",
        "\n",
        "Semakin besar nilai importance, semakin besar pengaruh fitur tersebut\n",
        "terhadap keputusan model.\n",
        "\"\"\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YP7Zts1SJ9dx",
        "outputId": "b9851c91-22c1-4693-f9ed-a258fef43071"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:37:19.049 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:19.050 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:19.053 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:19.054 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Footer"
      ],
      "metadata": {
        "id": "O3xcttT-KFYw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "st.markdown(\"---\")\n",
        "st.caption(\"UAS Data Science | Telco Customer Churn Prediction\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wmpbhHduKIDA",
        "outputId": "13807c7a-f510-47d1-e359-be79009e2723"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-12-25 15:37:46.983 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:46.985 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:46.987 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:46.990 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:46.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-12-25 15:37:46.992 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    }
  ]
}
