pipeline {
    agent any

    stages {
        stage("Checkout Code") {
            steps {
                git(url: 'https://github.com/dev-hack95/health_insurance_cross_sell_prediction', branch: 'dev-bac')
            }
        }

        stage("Check Logs") {
            steps {
                sh 'ls -la'
            }
        }

        stage("Check logs for pickle files") {
            steps {
                sh 'cd ./artifacts && ls -la'
            }
        }

        stage("Build") {
            steps {
                sh 'docker build -f ./Dockerfile . -t myapp:latest'
            }
        }

        stage("Run") {
            steps {
                sh 'docker-compose up -d'
            }
        }
    }
}