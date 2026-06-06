pipeline {
    agent any

    stages {

        stage('Clonar repositorio') {
            steps {
                echo 'Clonando el repositorio desde GitHub...'
                checkout scm
            }
        }

        stage('Verificar archivos del proyecto') {
            steps {
                echo 'Verificando estructura del proyecto...'
                sh 'ls -la'
                sh 'ls src/'
                sh 'ls notebooks/'
            }
        }

        stage('Verificar dependencias Python') {
            steps {
                echo 'Verificando que Python está disponible...'
                sh 'python3 --version'
                sh 'pip3 install numpy pandas scikit-learn matplotlib scipy --quiet'
            }
        }

        stage('Ejecutar preprocesamiento') {
            steps {
                echo 'Ejecutando preprocesamiento de datos...'
                sh 'python3 src/preprocessing.py'
            }
        }

        stage('Verificar resultados') {
            steps {
                echo 'Verificando archivos generados...'
                sh 'test -f data/processed/weather_clean.csv && echo "Dataset limpio generado correctamente"'
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado exitosamente.'
        }
        failure {
            echo 'El pipeline fallo. Revisar los logs.'
        }
    }
}