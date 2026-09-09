# IntentBot 🤖

A lightweight, modular Intent Classification Bot built with **C#** and **ML.NET** on **.NET 10**. 

This repository serves as a **Proof of Concept (PoC)** demonstrating how to implement natural language classification and supervised multi-class training directly within the .NET ecosystem—without relying on heavy Python runtimes or external web APIs.

## 🌟 Key Features

* **Multi-Class Classification**: Classifies natural language inputs into distinct operational intents (`Greeting`, `Time`, `About`, `Goodbye`).
* **Modular Architecture**: Clean separation of data contracts (`ModelInput`, `ModelOutput`) and pipeline execution logic.
* **On-Device ML Inference**: Trains and predicts entirely in-memory using `Microsoft.ML`.
* **Cross-Platform**: Developed and tested on **Linux (Manjaro)** via the .NET CLI.

## 🛠️ Tech Stack

* **Language**: C# 12
* **Framework**: 10.0 SDK
* **ML Engine**: `Microsoft.ML` (`SdcaMaximumEntropy` Classifier)
* **Data Format**: CSV vectorization

## 🚀 Getting Started

### Prerequisites
* [.NET 8.0 SDK](https://dotnet.microsoft.com/download) or higher installed.

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vlad-ux-png/IntentBot.git](https://github.com/Vlad-ux-png/IntentBot.git)
   cd IntentBot
   ```

2. **Restore dependencies:**
   ```bash
   dotnet restore
   ```

3. **Run the application:**
   ```bash
   dotnet run
   ```

## ⚙️ How It Works

1. **Data Loading**: Loads text-label pairs from `data.csv` into an `IDataView` mapped via `ModelInput`.
2. **Text Featurization**: Normalizes text and creates feature vectors using `FeaturizeText`.
3. **Model Training**: Trains a multi-class model with `SdcaMaximumEntropy` mapping features to `Label`.
4. **Inference**: Creates a `PredictionEngine` to classify user inputs in a CLI loop and map output via `ModelOutput`.
