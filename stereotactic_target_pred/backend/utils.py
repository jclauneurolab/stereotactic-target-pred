import logging
from traceback import format_exception

from flask import Flask, Response, jsonify
from flask_cors import CORS

local_devel = False

if local_devel:
    # ---Python---
    app = Flask(__name__)
    cors = CORS(app=app)

    #
    # Configure logger
    #
    app.logger.handlers.clear() # prevent double-logging with Flask logger
    log_handler = logging.StreamHandler() # log to stdout/stderr
    # Add logger to Flask app (only logs application errors, not HTTP errors)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s"
    )
    log_handler.setFormatter(log_formatter)
    log_handler.setLevel(logging.DEBUG) # this should allow all error messages to be displayed
    app.logger.addHandler(log_handler)
    # Use this logger for manual addition of log messages
    logger = logging.getLogger("werkzeug")
    logger.setLevel(logging.DEBUG)
else:
    app = Flask(
        __name__,
        static_folder = "../frontend/dist", # For local testing
        # static_folder="/app/stereotactic_target_pred/stereotactic_target_pred/backend/client",
        static_url_path="/"
    )
    cors = CORS(app=app)
    # Add logger to Flask app (only logs application errors, not HTTP errors)
    logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
    logging.debug("Static URL path is set to: /" + str(app.static_url_path))

def bad_request(message):
    """
    Return a 400 error with the given message a JSON.
    """
    response = jsonify({"error": message})
    response.status_code = 400
    return response

def server_error(message):
    """
    Return a 500 error with the given message as JSON.
    """
    response = jsonify({"error": message})
    response.status_code = 500
    return response

def bad_route(path):
    """
    Return a 404 error with the given path contained in the HTML.
    """
    response = f"""
    <!DOCTYPE HTML PUBLIC "-//W3C/DTD HTML 3.2 Final//EN">
    <title>404 Not Found</title>
    <h1>Not Found</h1>
    <p>The requested URL (path=/{path}) was not found on the server.
    If you entered the URL manually please check your spelling and try again.</p>
    """
    return Response(response, status=404, mimetype="text/html")

def log_tracebook(e):
    lines = format_exception(type(e), e, e.__traceback__)
    traceback = "".join(lines)
    logger.error(traceback)