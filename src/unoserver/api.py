from __future__ import annotations

import logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import os
import requests
import pydantic


from unoserver.client import UnoClient

logger = logging.getLogger("unoserver.api")


app = FastAPI(title="UnoConvert API", version="0.1")

class ConvertRequest(pydantic.BaseModel):
    fileurl: str
    filename: str
    convert_to: Optional[str] = 'pdf'
    filtername: Optional[str] = None
    filter_options: Optional[List[str]] = None
    update_index: bool = True
    infiltername: Optional[str] = None
    server: str = "127.0.0.1"
    port: str = "2003"
    protocol: str = "http"

@app.post("/convert2")
async def convert2(
    request: ConvertRequest
):
    """Convert an file using an existing `unoserver` instance.

    - fileurl: file url
    - filename: file name
    - convert_to: output file extension (eg. "pdf") — required when not passing an output filename
    """
    try:
        logger.info(f"[{request.filename}] Received conversion request change to {request.convert_to}: {request.fileurl}")
        response = requests.get(request.fileurl)

        client = UnoClient(server=request.server, port=request.port, host_location="remote", protocol=request.protocol)

        try:
            result = client.convert(
                indata=response.content,
                convert_to=request.convert_to,
                filtername=request.filtername,
                filter_options=request.filter_options or [],
                update_index=request.update_index,
                infiltername=request.infiltername,
            )
        except Exception as e:
            logger.exception(f"[{request.filename}] Conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        if result is None:
            # When server saved directly to disk, nothing to return
            logger.info(f"[{request.filename}] Conversion result saved on server")
            return JSONResponse({"detail": "Saved on server"})

        # Return the converted file as attachment
        filename = f"converted.{request.convert_to}"
        return StreamingResponse(
            iter([result]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException as e:
        logger.exception(f"[{filename}] Error processing conversion request: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"[{filename}] Error processing conversion request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    convert_to: Optional[str] = Form(None),
    filtername: Optional[str] = Form(None),
    filter_options: Optional[List[str]] = Form(None),
    update_index: bool = Form(True),
    infiltername: Optional[str] = Form(None),
    server: str = Form("127.0.0.1"),
    port: str = Form("2003"),
    protocol: str = Form("http"),
):
    """Convert an uploaded file using an existing `unoserver` instance.

    - file: uploaded file content
    - convert_to: output file extension (eg. "pdf") — required when not passing an output filename
    - filtername/infiltername/filter_options: forwarded to UnoClient
    - update_index: whether to update indexes before conversion
    - server/port/protocol: location for the RPC unoserver
    """
    filename = file.filename or "unknown"
    try:

        logger.info(f"[{filename}] Received conversion request: change to {convert_to}")
        content = await file.read()

        if convert_to is None:
            # Try to infer from filename
            _, ext = os.path.splitext(file.filename or "")
            if ext:
                convert_to = ext.lstrip('.').lower()
            else:
                raise HTTPException(status_code=400, detail="convert_to is required when filename has no extension")

        client = UnoClient(server=server, port=port, host_location="remote", protocol=protocol)

        try:
            result = client.convert(
                indata=content,
                convert_to=convert_to,
                filtername=filtername,
                filter_options=filter_options or [],
                update_index=update_index,
                infiltername=infiltername,
            )
        except Exception as e:
            logger.exception(f"[{filename}] Conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        if result is None:
            # When server saved directly to disk, nothing to return
            logger.info(f"[{filename}] Conversion result saved on server")
            return JSONResponse({"detail": "Saved on server"})

        # Return the converted file as attachment
        filename = f"converted.{convert_to}"
        return StreamingResponse(
            iter([result]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException as e:
        logger.exception(f"[{filename}] Error processing conversion request: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"[{filename}] Error processing conversion request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
def info(server: str = "127.0.0.1", port: str = "2003", protocol: str = "http"):
    """Return unoserver info (version, filters)."""
    client = UnoClient(server=server, port=port, host_location="auto", protocol=protocol)
    try:
        # Use the internal connection check which returns info
        with __import__("xmlrpc.client").client.ServerProxy(f"{protocol}://{server}:{port}", allow_none=True) as proxy:
            info = client._connect(proxy, retries=3, sleep=1)
            return info
    except Exception as e:
        logger.exception("Failed to get info from unoserver")
        raise HTTPException(status_code=502, detail=str(e))


def main():
    """Simple runner to start the API with uvicorn when invoked as a script."""
    import argparse
    import uvicorn
    import logging
    import os

    parser = argparse.ArgumentParser("unoserver-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Logging level for the API and uvicorn",
    )
    args = parser.parse_args()

    # Configure Python logging to match requested level. Uvicorn also accepts a
    # log_level parameter which controls server logs.
    try:
        level = getattr(logging, args.log_level.upper())
    except Exception:
        level = logging.INFO

    logging.basicConfig(level=level)

    uvicorn.run(
        "unoserver.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
