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

        stage('Instalar Python y dependencias') {
            steps {
                echo 'Instalando Python y dependencias...'
                sh 'apt-get update -q && apt-get install -y -q python3 python3-pip'
                sh 'pip3 install numpy pandas scikit-learn matplotlib scipy --quiet --break-system-packages'
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