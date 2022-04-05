#!/bin/bash
python manage.py init_db
#python manage.py run -h 0.0.0.0
exec "$@"