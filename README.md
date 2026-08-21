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
| [movies_ratings_eda](02_Data_Analysis_EDA/movies_ratings_eda.ipynb) | MovieLens 100k - genres, decades, Bayesian weighted top-10 |

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
Word2Vec · GloVe · FastText · SMS spam detection (TF) · Fake news detection (TF) · Naive Bayes text classification · **HuggingFace transformers basics** · **DistilBERT sentiment fine-tuning** · **LDA topic modeling** · **spaCy NER & text processing**

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
| Notebook | Topic |
|---|---|
| [spark_basics](07_BigData_Spark/spark_basics.ipynb) | Spark fundamentals |
| [spark_rdd_basics](07_BigData_Spark/spark_rdd_basics.ipynb) | RDD operations |
| [pyspark_dataframe](07_BigData_Spark/pyspark_dataframe.ipynb) | DataFrame API |
| [pyspark_mysql](07_BigData_Spark/pyspark_mysql.ipynb) | PySpark + MySQL |
| [pyspark_mllib_pipeline](07_BigData_Spark/pyspark_mllib_pipeline.ipynb) | ML pipelines, CrossValidator tuning |
| [spark_structured_streaming](07_BigData_Spark/spark_structured_streaming.ipynb) | Windows, watermarks, output modes |
| [spark_sql_window_functions](07_BigData_Spark/spark_sql_window_functions.ipynb) | Rankings, running totals, lag/lead |

## 🕸️ 08_Graphs_Networks
| Notebook | Topic |
|---|---|
| [network_graph](08_Graphs_Networks/network_graph.ipynb) | Graph visualization |
| [networkx_practical_applications](08_Graphs_Networks/networkx_practical_applications.ipynb) | NetworkX in practice |
| [graph_algorithms_centrality_community](08_Graphs_Networks/graph_algorithms_centrality_community.ipynb) | 5 centralities, paths, Louvain communities |
| [gnn_intro_pytorch_geometric](08_Graphs_Networks/gnn_intro_pytorch_geometric.ipynb) | GCN node classification on Cora |

## 🔬 09_Other_Experiments
Quantum computing with Qiskit ([base circuits](09_Other_Experiments/quantum/qiskit_base_circuit.ipynb), [Grover & teleportation](09_Other_Experiments/quantum/quantum_grover_teleportation.ipynb)) · [SciPy optimization: LP/MILP/curve-fitting](09_Other_Experiments/scipy_optimization_linear_programming.ipynb) · Pytensor demo · Java samples ([java/](09_Other_Experiments/java))

## 🎮 10_Reinforcement_Learning
| Notebook | Topic |
|---|---|
| [q_learning_intro](10_Reinforcement_Learning/q_learning_intro.ipynb) | Tabular Q-learning on FrozenLake |
| [dqn_cartpole_pytorch](10_Reinforcement_Learning/dqn_cartpole_pytorch.ipynb) | Deep Q-network: replay buffer + target net |
| [ppo_stable_baselines3_intro](10_Reinforcement_Learning/ppo_stable_baselines3_intro.ipynb) | PPO policy gradients with SB3 |

## 🚀 11_MLOps_Deployment
| Notebook | Topic |
|---|---|
| [streamlit_model_app](11_MLOps_Deployment/streamlit_model_app.ipynb) | Model → interactive web app |
| [fastapi_model_serving](11_MLOps_Deployment/fastapi_model_serving.ipynb) | REST API + pydantic validation |
| [docker_for_ml](11_MLOps_Deployment/docker_for_ml.ipynb) | Containerize a model service |
| [github_actions_ci_for_ml](11_MLOps_Deployment/github_actions_ci_for_ml.ipynb) | CI quality gates for models |
| [mlflow_experiment_tracking](11_MLOps_Deployment/mlflow_experiment_tracking.ipynb) | Params/metrics/artifacts tracking |
| [model_monitoring_drift_evidently](11_MLOps_Deployment/model_monitoring_drift_evidently.ipynb) | Data drift detection & reports |

---
### Libraries
NumPy · Pandas · Matplotlib · Seaborn · Plotly · SciPy · scikit-learn · XGBoost · Prophet · statsmodels · TensorFlow/Keras · PyTorch · HuggingFace Transformers · diffusers · Whisper · LangChain · ChromaDB/FAISS · Ollama · Spark · Gymnasium · Streamlit · FastAPI · MLflow
