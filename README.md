# AWS Amazon Recommender

**Author**: Sameera Boppana

## Overview

The "AWS Amazon Recommender" project is focused on developing a recommendation system using AWS services to enhance the user experience on the Amazon platform. This system leverages advanced machine learning techniques and cloud-based infrastructure to deliver personalized product recommendations to users based on their browsing and purchasing history. The project explores the use of AWS tools such as SageMaker, Lambda, and DynamoDB to build, train, and deploy the recommendation engine at scale.

### Amazon User Recommender App

The Amazon User Recommender App is designed to train a content-based and a collaborative filter recommender system to provide the top 10 product predictions for a specified user. The application is dockerized and runs on Amazon Elastic Container Service (ECS) and EC2, with all artifacts stored in an S3 bucket.

## Table of Contents

- [Project Objective](#project-objective)
- [Tech Stack](#tech-stack)
- [Data Collection and Processing](#data-collection-and-processing)
- [Modeling Approaches](#modeling-approaches)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Content-Based Filtering](#content-based-filtering)
  - [Hybrid Approach](#hybrid-approach)
- [AWS Implementation](#aws-implementation)
  - [AWS SageMaker](#aws-sagemaker)
  - [AWS Lambda](#aws-lambda)
  - [AWS DynamoDB](#aws-dynamodb)
- [Amazon User Recommender App](#amazon-user-recommender-app)
  - [Application Components](#application-components)
  - [Setup Instructions](#setup-instructions)
- [Results](#results)
- [Deployment](#deployment)
- [Challenges and Future Work](#challenges-and-future-work)
- [How to Use](#how-to-use)
- [Repository Structure](#repository-structure)
- [Contributors](#contributors)

## Project Objective

The main objective of this project is to build a scalable, efficient, and accurate recommendation system using AWS services that can enhance the shopping experience for Amazon users. The recommender system is designed to predict and suggest products to users based on their past behavior and preferences, improving user satisfaction and increasing sales.

## Tech Stack

- **Programming Language**: Python
- **Cloud Platform**: Amazon Web Services (AWS)

### Key AWS Services:

- **AWS SageMaker**: For building, training, and deploying machine learning models.
- **AWS Lambda**: For running code in response to events and integrating various services.
- **AWS DynamoDB**: For storing and managing recommendation data.
- **AWS S3**: For data storage and retrieval.
- **AWS IAM**: For managing access to AWS resources.

## Data Collection and Processing

### Data Sources:

- **User Interaction Data**: Includes browsing history, past purchases, and ratings provided by users.
- **Product Metadata**: Detailed information about products, including category, price, description, and more.
- **User Profiles**: Demographic and behavioral data about users to enhance personalization.

### Data Processing:

- **Data Cleaning**: Removal of incomplete or inconsistent data entries to ensure high-quality input for the model.
- **Feature Engineering**: Creation of additional features such as user-product interactions, time-based behavior, and more.
- **Data Normalization**: Standardizing data formats and scales to improve model performance.

## Modeling Approaches

### Collaborative Filtering

- **User-Based Filtering**: Recommends products based on similar users' preferences.
- **Item-Based Filtering**: Suggests products that are similar to items the user has interacted with in the past.

### Content-Based Filtering

- **Feature Extraction**: Uses product metadata to recommend items with similar characteristics to those previously liked by the user.
- **Text Analysis**: Natural language processing techniques applied to product descriptions and reviews to identify relevant items.

### Hybrid Approach

- **Combined Methods**: Utilizes both collaborative and content-based filtering to enhance recommendation accuracy.
- **Model Stacking**: Multiple models are trained and their predictions are combined to produce the final recommendation.

## AWS Implementation

### AWS SageMaker

- **Model Training**: SageMaker is used to train collaborative filtering and content-based models on large datasets.
- **Hyperparameter Tuning**: Automated tuning processes are employed to optimize model performance.
- **Model Hosting**: Deploy the trained models using SageMaker endpoints to make real-time recommendations.

### AWS Lambda

- **Event-Driven Execution**: AWS Lambda functions are used to trigger recommendation updates based on user activity, such as viewing a product or making a purchase.
- **Integration**: Lambda functions integrate various AWS services, ensuring smooth data flow and processing.

### AWS DynamoDB

- **Data Storage**: DynamoDB stores user profiles, interaction data, and recommendation results for quick retrieval.
- **Scalability**: DynamoDB’s scalability ensures that the recommender system can handle large volumes of data and requests.

## Amazon User Recommender App

### Application Components

The application consists of the following components:

- **Streamlit Application**: Allows the user to select a recommender system and input a user ID.
- **Prediction Display**: Based on the input and selected model, a prediction is made and displayed to the application.

### Setup Instructions

To set up this project, follow these steps:

1. **Clone the Repository**: Clone the Clouds Data Pipeline repository to your local machine.

    ```bash
    git clone https://github.com/DarwinYip2022/Cloud_Engineering.git
    ```

2. **Install Requirements**: Install the required Python packages using pip.

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables**: Create a `.env` file to securely configure AWS S3 credentials.

    ```env
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_REGION=
    ```

4. **Run the Python Application for Model Training**:

    ```bash
    python3 pipeline.py --config config/default.yaml
    ```

5. **Run the Streamlit Application**:

    ```bash
    streamlit run app.py
    ```

### Build the Application Docker Image

```bash
docker build -t amazon-app .
```

### Run the Entire Model Pipeline in a Docker Container 
```bash
docker run -p 8501:8501 --env-file .env amazon-app
```
