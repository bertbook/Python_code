from flask import Flask, request
import json

from TextSummarization.TextSummarization import text_summary

app=Flask(__name__)


@app.route ("/textSummarization", methods=['POST'])
def textClassification ():
    try:
        json_data = request.get_json(force=True)
        input_text = json_data['query']
        summary = text_summary(input_text)

        result = dict()
        result['query'] = input_text
        result['summary'] = summary

        result = json.dumps(result)
        return result

    except Exception as e:
        error = {"Error": str(e)}
        error = json.dumps(error)
        return error


if __name__ == "__main__" :
    app.run(port="5000")
