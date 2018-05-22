from apscheduler.schedulers.background import BackgroundScheduler
from scraping import do
from flask import Flask

def sensor():
    do()
    print("Scheduler is alive :)!")

sched = BackgroundScheduler(daemon=True)
app = Flask(__name__)


sched.add_job(sensor,'cron', day_of_week='mon-sun', hour=7)
sched.start()

@app.route('/')
def hello():
    return "Hello:) !"


if __name__ == "__main__":
    app.run()






