# NeuralNetwork-Visualization

## Run Frontend
~~~
cd playground
npm i
npm run build
npm run serve
~~~

## Run Backend
~~~
cd Backend/VisProject
python manage.py migrate
python manage.py runserver
cd ..
python train.py
~~~

## Rerun the server
~~~
python manage.py makemigrations
python manage.py migrate
~~~

