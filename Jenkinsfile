pipeline {
    agent any

    stages {

        stage('Clonar repositorio') {
            steps {
                echo 'Clonando el repositorio desde GitHub...'
                checkout scm
            }
        }

        stage('Verificar estructura del proyecto') {
            steps {
                echo 'Verificando archivos del proyecto...'
                sh 'ls -la'
                sh 'ls src/'
                sh 'ls notebooks/'
                sh 'ls data/'
            }
        }

        stage('Verificar Dockerfile') {
            steps {
                echo 'Verificando configuracion Docker...'
                sh 'cat Dockerfile'
            }
        }

        stage('Verificar configuracion PSO-ANFIS') {
            steps {
                echo 'Verificando modulos del sistema...'
                sh 'cat src/anfis_model.py | head -20'
                sh 'cat src/pso.py | head -20'
            }
        }
    }

    post {
        success {
            echo 'Pipeline verificado exitosamente. Proyecto listo para ejecutar.'
        }
        failure {
            echo 'El pipeline fallo. Revisar los logs.'
        }
    }
}