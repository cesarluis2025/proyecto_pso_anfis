pipeline {
    agent any

    stages {

        stage('Clonar repositorio') {
            steps {
                echo 'Clonando el repositorio desde GitHub...'
                checkout scm
            }
        }

        stage('Construir imagen Docker') {
            steps {
                echo 'Construyendo la imagen Docker del proyecto...'
                sh 'docker-compose build'
            }
        }

        stage('Ejecutar pipeline PSO-ANFIS') {
            steps {
                echo 'Ejecutando el modelo PSO-ANFIS dentro del contenedor...'
                sh 'docker-compose run --rm jupyter python main.py'
            }
        }

        stage('Verificar resultados') {
            steps {
                echo 'Verificando que se generaron los archivos de resultados...'
                sh 'test -f results/tables/comparativa.csv && echo Tabla generada correctamente'
                sh 'test -f results/figures/fig4_convergencia_pso.png && echo Graficas generadas correctamente'
            }
        }

        stage('Limpiar contenedor') {
            steps {
                echo 'Limpiando contenedores Docker...'
                sh 'docker-compose down'
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