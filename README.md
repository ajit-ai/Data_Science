# Data_Science

A curated collection of Jupyter notebook projects and tutorials covering data science,
machine learning, deep learning, NLP, GenAI/RAG, big data, and graph analysis.

Datasets live in [`content/`](content/).
Library setup: `pip install -r requirements.txt` · Most notebooks also run free on [Google Colab](https://colab.research.google.com) via the badges below.

## 📚 01_Fundamentals
Core library tutorials.

| Notebook | Topic |
|---|---|
| [numpy_pandas_matplotlib_seaborn_sklearn](01_Fundamentals/numpy_pandas_matplotlib_seaborn_sklearn.ipynb) | NumPy, Pandas, Matplotlib, Seaborn, scikit-learn crash course |
| [data_visualization_with_seaborn](01_Fundamentals/data_visualization_with_seaborn.ipynb) | Seaborn visualization guide |
| [astropy_tutorials](01_Fundamentals/astropy_tutorials.ipynb) | Astropy (astronomy computing) |
| [pandas/](01_Fundamentals/pandas) | DataFrame basics, operations, joins |
| [statistics_hypothesis_testing](01_Fundamentals/statistics_hypothesis_testing.ipynb) | Distributions, CLT, A/B tests, t-test, chi-squared |
| [plotly_interactive_visualization](01_Fundamentals/plotly_interactive_visualization.ipynb) | Interactive charts, maps, animation frames |

## 📊 02_Data_Analysis_EDA
Exploratory data analysis mini projects.

| Notebook | Topic |
|---|---|
| [covid-19/](02_Data_Analysis_EDA/covid-19) | Covid vaccines, Plotly Express visualization, country vaccination data |
| [ipl/ipl_data_analysis](02_Data_Analysis_EDA/ipl/ipl_data_analysis.ipynb) | IPL matches EDA |
| [zomato_data_analysis](02_Data_Analysis_EDA/zomato_data_analysis.ipynb) | Zomato restaurant data |
| [turing_data_analysis_test](02_Data_Analysis_EDA/turing_data_analysis_test.ipynb) | Cardio/covid analytical test |

## 🤖 03_Machine_Learning

### algorithms — single-concept demos
Linear & logistic regression · Decision tree · Random forest · **XGBoost/gradient boosting** · SVM & kernels · Naive Bayes ×2 · Clustering (intro, DBSCAN, types) · **PCA** · **Ridge/Lasso/ElasticNet** · **Hyperparameter tuning (Grid/Random/Optuna)** · Anomaly detection · Imbalanced data classification · Model comparison · ANN search · Backpropagation · KNN ([knn/](03_Machine_Learning/algorithms/knn)) · Gradient descent ([gradient_descent.py](03_Machine_Learning/algorithms/gradient_descent.py))

### projects — end-to-end predictions
- **healthcare/** — heart disease ×2, breast cancer ×2, cancer cells, Parkinson's, autism, disease prediction, calories burnt
- **finance-stocks/** — stock price ×4 (ML/SVM/TensorFlow/Microsoft), Dogecoin, portfolio optimization, credit card fraud
- **forecasting/** — sales forecast, Ola bike demand, vehicle count, rainfall, IPL score, **SARIMA**, **Prophet**
- **other/** — wine quality, house & real-estate prices, music popularity, customer churn, recommender systems ×2, biometric authentication (GenAI)

## 🧠 04_Deep_Learning
| Subfolder | Contents |
|---|---|
| tensorflow-basics | TF basics, examples, neural networks |
| pytorch | PyTorch fundamentals |
| fastai | fastai banana image classifier |
| cnn-rnn-lstm | CNN, RNN, LSTM ×3, NN classification |
| [transformers](04_Deep_Learning/transformers) | Attention from scratch (NumPy) - QKV, causal mask, multi-head, positional encoding |
| [computer-vision](04_Deep_Learning/computer-vision) | YOLOv8 object detection · ResNet18 transfer learning |
| generative-models | GAN, autoencoders, RBM |

## 💬 05_NLP_Embeddings
Word2Vec · GloVe · FastText · SMS spam detection (TF) · Fake news detection (TF) · Naive Bayes text classification · **HuggingFace transformers basics** · **DistilBERT sentiment fine-tuning**

## ✨ 06_GenAI_LLM_RAG
All runnable free on Google Colab - no paid API keys.

| Notebook | Topic |
|---|---|
| [rag_basics](06_GenAI_LLM_RAG/rag_basics.ipynb) | Retrieval-augmented generation fundamentals |
| [rag_langchain](06_GenAI_LLM_RAG/rag_langchain.ipynb) | RAG with LangChain |
| [llm_for_python](06_GenAI_LLM_RAG/llm_for_python.ipynb) | LLM APIs for Python |
| [pandasai](06_GenAI_LLM_RAG/pandasai.ipynb) | Conversational dataframes |
| [prompt_engineering](06_GenAI_LLM_RAG/prompt_engineering.ipynb) | Zero/few-shot, CoT, roles, JSON output |
| [llm_function_calling_agents](06_GenAI_LLM_RAG/llm_function_calling_agents.ipynb) | Tool use + ReAct agent loop |
| [sentence_transformers_embeddings](06_GenAI_LLM_RAG/sentence_transformers_embeddings.ipynb) | Semantic search with SBERT |
| [faiss_chroma_vector_db](06_GenAI_LLM_RAG/faiss_chroma_vector_db.ipynb) | ChromaDB + FAISS vector stores |
| [rag_evaluation_ragas](06_GenAI_LLM_RAG/rag_evaluation_ragas.ipynb) | Faithfulness, context precision/recall |
| [ollama_local_llm](06_GenAI_LLM_RAG/ollama_local_llm.ipynb) | Local Llama in Colab via Ollama |
| [stable_diffusion_text_to_image](06_GenAI_LLM_RAG/stable_diffusion_text_to_image.ipynb) | Diffusers text-to-image |
| [whisper_speech_recognition](06_GenAI_LLM_RAG/whisper_speech_recognition.ipynb) | Local speech-to-text |

## ⚡ 07_BigData_Spark
Spark basics · Spark RDDs · PySpark DataFrames · PySpark + MySQL

## 🕸️ 08_Graphs_Networks
Network graphs · NetworkX practical applications

## 🔬 09_Other_Experiments
Quantum computing with Qiskit · Pytensor demo · Java samples ([java/](09_Other_Experiments/java))

## 🎮 10_Reinforcement_Learning
[Q-learning intro](10_Reinforcement_Learning/q_learning_intro.ipynb) - tabular Q-learning on FrozenLake: exploration/exploitation, Bellman updates, greedy policy evaluation.

## 🚀 11_MLOps_Deployment
| Notebook | Topic |
|---|---|
| [streamlit_model_app](11_MLOps_Deployment/streamlit_model_app.ipynb) | Model → interactive web app |
| [fastapi_model_serving](11_MLOps_Deployment/fastapi_model_serving.ipynb) | REST API + pydantic validation + Docker |
| [mlflow_experiment_tracking](11_MLOps_Deployment/mlflow_experiment_tracking.ipynb) | Params/metrics/artifacts + run comparison UI |

---
### Libraries
NumPy · Pandas · Matplotlib · Seaborn · Plotly · SciPy · scikit-learn · XGBoost · Prophet · statsmodels · TensorFlow/Keras · PyTorch · HuggingFace Transformers · diffusers · Whisper · LangChain · ChromaDB/FAISS · Ollama · Spark · Gymnasium · Streamlit · FastAPI · MLflow
