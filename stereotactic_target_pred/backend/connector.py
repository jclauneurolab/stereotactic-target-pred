import re 

from flask import abort, request, send_from_directory

from utils import app, cors, bad_route, logger
from stereotaxy_route import post_model_predict_json, get_download_output_json

if __name__ != "__main__":
    logger.debug("Assuming app is configured for gunicorn in Docker container.")

    @app.route("/")
    def index():
        """
        Serve the index.html file when client requests the static url (e.g., <http://localhost:5000/>).
        """

        logger.info(f"Serving index.html because client requested {app.static_url_path}/")
        return app.send_static_file("index.html")
    
@app.route("/", defaults={"path": ""})
@app.route("/<string:path>", methods=["GET", "POST"])
@app.route("/<path:path>", methods=["GET", "POST"])
def redirect(path):
    """
    Redirects request to the appropriate function based on the path.
    """

    logger.info(f"Parsing request for /{path}")
    logger.info(f"URL of request: " + str(request.url))

    if re.search(r"\bdownload-output\b", path) is not None:
        logger.info("Redirecting request to /download-output")
        if request.method != "GET":
            abort(405)

        return get_download_output_json()
    
    elif re.search(r"\bapply-model\b", path) is not None:
        logger.info("Redirecting request to /apply-model")
        if request.method != "POST":
            abort(405)

        return post_model_predict_json()
    
    elif re.search(r"\bmanifest.json\b", path) is not None:
        logger.info("Serving manifest.json")
        if request.method != "GET":
            abort(405)

        return app.send_static_file("manifest.json")
    
    elif re.search(r"\bfavicon.ico\b", path) is not None:
        logger.info("Serving favicon.ico")
        if request.method != "GET":
            abort(405)

        return app.send_static_file("favicon.ico")
    
    elif re.search(r"\brobots.txt\b", path) is not None:
        logger.info("Serving robots.txt")
        if request.method != "GET":
            abort(405)
        
        return app.send_static_file("robots.txt")
    
    elif re.search(r"\bsitemap.xml\b", path) is not None:
        logger.info("Serving sitemap.xml")
        if request.method != "GET":
            abort(405)

        return app.send_static_file("sitemap.xml")
    
    else:
        filename = re.search(r"[^/?]*\.(?:gif|png|jpeg|jpg|ico|js|css)$", path)
        if app.static_folder is not None and filename is not None:
            if request.method != "GET":
                abort(405)
            logger.debug(f"Serving some image/js/css at /{path}")
            return send_from_directory(app.static_folder, filename.group())
        else:
            logger.error(f"Bad route: route=/{path}")
            return bad_route(path)


if __name__== "__main__":
    cors.init_app(app)
    app.run(host="0.0.0.0",port=5000, debug=True)
    
    