from flask import Flask, render_template, request, jsonify
import fmp_multi as fmp
import os

app = Flask(__name__)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        function_type = request.form.get('function_type')
        symbol = request.form.get('symbol', '').upper()
        period = request.form.get('period', '30min')

        if function_type == "Intraday Data":
            data = fmp.fmp_intra(symbol, period=period)
            if data:
                return jsonify({'success': True, 'data': data})
            return jsonify({'success': False, 'error': 'No intraday data found for the given symbol.'})

        elif function_type == "Company Profile":
            data = fmp.fmp_profF(symbol)
            if data:
                return jsonify({
                    'success': True,
                    'data': data
                })
            return jsonify({'success': False, 'error': 'No profile data found for the given symbol.'})

        elif function_type == "Search Data":
            data = fmp.fmp_search(symbol)
            if data:
                return jsonify({'success': True, 'data': data})
            return jsonify({'success': False, 'error': 'No search results found for the given term.'})

        elif function_type == "Earnings Dates":
            data = fmp.fmp_earnSym(symbol)
            if data:
                return jsonify({'success': True, 'data': data})
            return jsonify({'success': False, 'error': 'No earnings data found for the given symbol.'})

        elif function_type == "Dividends":
            data = fmp.fmp_div(symbol)
            if data:
                return jsonify({'success': True, 'data': data})
            return jsonify({'success': False, 'error': 'No dividend data found for the given symbol.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
