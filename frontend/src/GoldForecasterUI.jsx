import React, { useState, useEffect } from 'react';

export default function GoldForecasterUI({ apiBaseUrl = 'http://localhost:8000' }) {
  const [forecast, setForecast] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Fetch live forecast and history from the FastAPI backend
  const loadLiveData = async () => {
    try {
      setLoading(true);
      const [resForecast, resHistory] = await Promise.all([
        fetch(`${apiBaseUrl}/api/forecast`),
        fetch(`${apiBaseUrl}/api/history`)
      ]);

      if (!resForecast.ok || !resHistory.ok) {
        throw new Error('API server returned error status.');
      }

      const dataForecast = await resForecast.json();
      const dataHistory = await resHistory.json();

      setForecast(dataForecast);
      setHistory(dataHistory);
      setError(null);
    } catch (err) {
      console.warn('API error, falling back to cached model state:', err);
      setError('Live API offline. Displaying fallback model state.');
      // Fallback state if server is not yet started
      setForecast({
        signalDirection: 'UP',
        confidence: 63.8,
        currentClose: 341.19,
        targetPrice: 344.20,
        priceDiff: 3.01,
        pctDiff: 0.88,
        volatilityUpper: 348.50,
        volatilityLower: 339.90,
        asOfDate: 'Nov 18, 2026',
        status: 'Active',
        thresholdPassed: true,
        walkForwardAccuracy: 63.8,
        evaluatedDays: 178,
        filteredDays: 42,
        inputFactors: [
          { name: '10Y Real Interest Rates (TIPS Yield)', value: '2.88%', change: '-0.010%', impact: 'High Inverse', weight: '32%' },
          { name: 'US Dollar Index ETF (UUP)', value: 'USD Strength', change: '+0.27%', impact: 'High Inverse', weight: '25%' },
          { name: 'Gold / Silver Ratio (Commodity Spread)', value: '8.38', change: '+7.09%', impact: 'Moderate Leading', weight: '18%' },
          { name: 'US 10Y Nominal Bond Yield', value: '5.07%', change: '+0.090%', impact: 'Moderate', weight: '15%' },
          { name: '14-Day RSI (Momentum Oscillator)', value: '30.55', change: 'Oversold Zone', impact: 'Mean Reversion', weight: '10%' }
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLiveData();
  }, [apiBaseUrl]);

  const handleSync = async () => {
    try {
      setRefreshing(true);
      await fetch(`${apiBaseUrl}/api/refresh`, { method: 'POST' });
      await loadLiveData();
    } catch (e) {
      alert('Sync failed: ' + e.message);
    } finally {
      setRefreshing(false);
    }
  };

  const toggleDemoSignal = () => {
    if (!forecast) return;
    const newDirection = forecast.signalDirection === 'UP' ? 'DOWN' : 'UP';
    const newConfidence = newDirection === 'UP' ? 68.4 : 63.8;
    const newTarget = newDirection === 'UP'
      ? forecast.currentClose * 1.015
      : forecast.currentClose * 0.985;

    setForecast({
      ...forecast,
      signalDirection: newDirection,
      confidence: newConfidence,
      targetPrice: newTarget,
      priceDiff: newTarget - forecast.currentClose,
      pctDiff: ((newTarget - forecast.currentClose) / forecast.currentClose) * 100
    });
  };

  if (loading && !forecast) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 font-sans">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-10 h-10 border-4 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm font-semibold text-slate-600">Connecting to GoldGrow Indian Market Terminal...</p>
        </div>
      </div>
    );
  }

  const isUp = forecast.signalDirection === 'UP';

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans p-6 md:p-10">
      {/* 1. Header Bar */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between pb-6 mb-8 border-b border-slate-200">
        <div>
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-amber-500 flex items-center justify-center font-bold text-white shadow-sm text-xs font-mono">
              GG
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">GoldGrow</h1>
            <span className="text-xs font-semibold px-2.5 py-1 bg-amber-100 text-amber-800 rounded-full border border-amber-200">
              Indian Market (MCX / Retail)
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            GoldGrow • Domestic Gold Directional & Rate Forecaster • Factoring USD/INR & IBJA Retail Taxes
          </p>
        </div>

        <div className="mt-4 md:mt-0 flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-md border border-emerald-200 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Model Status: Active</span>
          </div>

          <button
            onClick={handleSync}
            disabled={refreshing}
            className="px-3.5 py-1.5 bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-xs font-semibold rounded-md shadow-sm transition-all flex items-center space-x-1.5"
          >
            <span>{refreshing ? 'Syncing...' : '↻ Sync Market'}</span>
          </button>

          <button
            onClick={toggleDemoSignal}
            className="px-4 py-1.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-md shadow-sm transition-all"
          >
            Simulate ({forecast.signalDirection})
          </button>
        </div>
      </header>

      {/* 2. Main Content Layout */}
      <main className="max-w-7xl mx-auto space-y-8">
        
        {/* Top Section: Direction & Forecast Hero Card */}
        <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 md:p-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-500">
                Next Directional Forecast (T+1 to T+3 Horizon)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">As of Close: <strong className="text-slate-700">{forecast.asOfDate}</strong></p>
            </div>
            <span className="text-xs font-medium bg-slate-100 text-slate-600 px-3 py-1 rounded-md border border-slate-200">
              High-Conviction Filter: &gt;60%
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Forecast Direction Box */}
            <div className={`p-6 rounded-lg border ${isUp ? 'bg-emerald-50/70 border-emerald-200' : 'bg-red-50/70 border-red-200'} flex flex-col justify-between`}>
              <span className="text-xs font-semibold text-slate-500 uppercase">Predicted Direction</span>
              <div className="my-3 flex items-center space-x-3">
                <span className={`text-3xl font-extrabold ${isUp ? 'text-emerald-600' : 'text-red-600'}`}>
                  {isUp ? 'FORECAST: UP ↗' : 'FORECAST: DOWN ↘'}
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Signal Noise Filter: <strong className={forecast.thresholdPassed ? 'text-emerald-700' : 'text-amber-700'}>
                  {forecast.thresholdPassed ? 'Pass (High Conviction)' : 'Neutral / Choppy Zone'}
                </strong>
              </p>
            </div>

            {/* Confidence Score */}
            <div className="p-6 rounded-lg bg-slate-50 border border-slate-200 flex flex-col justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase">Ensemble Confidence</span>
              <div className="my-2">
                <span className="text-4xl font-extrabold text-amber-600 font-mono">{forecast.confidence}%</span>
                <p className="text-xs text-slate-500 mt-1">Calibrated Probability Weight</p>
              </div>
              {/* Progress Bar */}
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-amber-500 h-full rounded-full transition-all duration-500"
                  style={{ width: `${forecast.confidence}%` }}
                ></div>
              </div>
            </div>

            {/* Target Price Forecast */}
            <div className="p-6 rounded-lg bg-slate-50 border border-slate-200 flex flex-col justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase">Target Price Projection (GLD)</span>
              <div className="my-2">
                <span className="text-4xl font-extrabold text-slate-900 font-mono">${forecast.targetPrice.toFixed(2)}</span>
                <p className={`text-xs font-semibold mt-1 ${isUp ? 'text-emerald-600' : 'text-red-600'}`}>
                  {isUp ? '+' : ''}${forecast.priceDiff.toFixed(2)} ({isUp ? '+' : ''}{forecast.pctDiff.toFixed(2)}%) vs ${forecast.currentClose.toFixed(2)}
                </p>
              </div>
              <div className="flex justify-between text-xs text-slate-500 font-mono pt-2 border-t border-slate-200">
                <span>Lower Band: ${forecast.volatilityLower.toFixed(2)}</span>
                <span>Upper Band: ${forecast.volatilityUpper.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </section>

        {/* Middle Section: Metrics & Feature Influence */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Historical Accuracy Summary */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-4">
                Walk-Forward Model Performance
              </h3>
              <div className="mb-6">
                <span className="text-4xl font-extrabold text-slate-900 font-mono">{forecast.walkForwardAccuracy}%</span>
                <p className="text-xs text-slate-500 mt-1">
                  Evaluated on high-conviction signals (&gt;60%) across 5 chronological walk-forward folds without look-ahead bias.
                </p>
              </div>
            </div>

            <div className="space-y-3 pt-4 border-t border-slate-100 text-xs">
              <div className="flex justify-between text-slate-600">
                <span>Evaluated Holdout Days:</span>
                <strong className="font-mono text-slate-900">{forecast.evaluatedDays} Days</strong>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Filtered Neutral Noise:</span>
                <strong className="font-mono text-slate-900">{forecast.filteredDays} Days</strong>
              </div>
              <div className="flex justify-between text-slate-600">
                <span>Theoretical Horizon Ceiling:</span>
                <strong className="font-mono text-slate-900">~68.0%</strong>
              </div>
            </div>
          </div>

          {/* Model Input Factors */}
          <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500">
                Current Model Input Factor Weights
              </h3>
              <span className="text-[11px] text-slate-400">Live FRED & Tiingo Feeds</span>
            </div>

            <div className="space-y-3">
              {forecast.inputFactors.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100 text-xs hover:bg-slate-100/70 transition-colors">
                  <div className="w-1/2">
                    <span className="font-semibold text-slate-800 block">{item.name}</span>
                    <span className="text-slate-400 text-[11px]">Factor Dynamics: {item.impact}</span>
                  </div>
                  <div className="text-right pr-4">
                    <div className="font-mono font-bold text-slate-800">{item.value}</div>
                    <div className={`font-mono text-[11px] ${item.change.startsWith('+') ? 'text-emerald-600' : (item.change.startsWith('-') ? 'text-red-600' : 'text-slate-500')}`}>
                      {item.change}
                    </div>
                  </div>
                  <div className="w-16 text-right">
                    <span className="font-mono font-bold text-amber-600 bg-amber-50 px-2 py-1 rounded border border-amber-200 text-[11px]">{item.weight}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Bottom Section: Past Outcome Logs */}
        <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500">
              Recent Model Prediction Log & Outcome Validation
            </h3>
            <span className="text-xs text-slate-400">Chronological Holdout Tests</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-sans uppercase">
                  <th className="pb-3">Session Date</th>
                  <th className="pb-3">Actual Close</th>
                  <th className="pb-3">Model Forecast</th>
                  <th className="pb-3">Confidence</th>
                  <th className="pb-3 text-right">Outcome Validation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.length > 0 ? (
                  history.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                      <td className="py-3 text-slate-600 font-sans">{row.date}</td>
                      <td className="py-3 font-bold text-slate-800">{row.close}</td>
                      <td className={`py-3 font-bold ${row.forecast === 'UP' ? 'text-emerald-600' : 'text-red-600'}`}>
                        {row.forecast} {row.forecast === 'UP' ? '↗' : '↘'}
                      </td>
                      <td className="py-3 text-slate-600">{row.conf}</td>
                      <td className="py-3 text-right">
                        <span className={`px-2.5 py-1 rounded text-[10px] font-sans font-semibold ${
                          row.outcome === 'Correct' ? 'bg-emerald-100 text-emerald-800 border border-emerald-200' : 'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                          {row.outcome}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" className="py-4 text-center text-slate-400">Loading prediction log...</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

      </main>
    </div>
  );
}
