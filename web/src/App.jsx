import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, ImageClassifier } from '@mediapipe/tasks-vision'
import './App.css'

const UMBRAL = 0.6
const ROI = { left: 0.2, top: 0.1, right: 0.8, bottom: 0.9 }
const FRAMES_CONFIRMAR = 15
const FRAMES_LIBERAR = 20

const ETIQUETAS = {
  plastico: 'Plástico',
  papel: 'Papel',
  carton: 'Cartón',
  metal: 'Metal',
  vidrio: 'Vidrio',
  organico: 'Orgánico',
}

const RECOMENDACIONES = {
  plastico: 'Contenedor de plásticos. Enjuaga y aplasta el envase antes de depositarlo.',
  papel: 'Contenedor de papel. Debe estar limpio y seco, sin restos de grasa.',
  carton: 'Contenedor de cartón. Aplana las cajas y quita cintas y grapas.',
  metal: 'Contenedor de metales. Enjuaga las latas y aplástalas.',
  vidrio: 'Contenedor de vidrio. Vacía y enjuaga el envase; no lo mezcles con cerámica ni espejos.',
  organico: 'Contenedor de orgánicos o composta. Deposítalo sin bolsas de plástico.',
}

const CONTEO_INICIAL = {
  plastico: 0,
  papel: 0,
  carton: 0,
  metal: 0,
  vidrio: 0,
  organico: 0,
}

export default function App() {
  const videoRef = useRef(null)
  const seguimiento = useRef({ cat: null, racha: 0, vacios: 0, contada: null })
  const [estado, setEstado] = useState('Cargando modelo...')
  const [scores, setScores] = useState([])
  const [conteo, setConteo] = useState(CONTEO_INICIAL)

  useEffect(() => {
    let cancelado = false
    let classifier = null
    let stream = null
    let rafId = 0
    let ultimoTiempo = -1

    const registrar = (top) => {
      const s = seguimiento.current
      const cat = top && top.score >= UMBRAL ? top.categoryName : null

      if (cat === null) {
        s.cat = null
        s.racha = 0
        s.vacios += 1
        if (s.vacios >= FRAMES_LIBERAR) s.contada = null
        return
      }

      s.vacios = 0
      if (cat !== s.cat) {
        s.cat = cat
        s.racha = 1
        return
      }

      s.racha += 1
      if (s.racha >= FRAMES_CONFIRMAR && s.contada !== cat) {
        s.contada = cat
        setConteo((c) => ({ ...c, [cat]: c[cat] + 1 }))
      }
    }

    async function iniciar() {
      try {
        const vision = await FilesetResolver.forVisionTasks('/wasm')
        classifier = await ImageClassifier.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/modelo/model.tflite' },
          runningMode: 'VIDEO',
          maxResults: 6,
        })
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
        if (cancelado) {
          stream.getTracks().forEach((t) => t.stop())
          classifier.close()
          return
        }
        const video = videoRef.current
        video.srcObject = stream
        await video.play()
        setEstado('Listo')

        const bucle = () => {
          if (cancelado) return
          if (video.currentTime !== ultimoTiempo) {
            ultimoTiempo = video.currentTime
            const res = classifier.classifyForVideo(video, performance.now(), {
              regionOfInterest: ROI,
            })
            const cats = res.classifications[0]?.categories ?? []
            const ordenadas = [...cats].sort((a, b) => b.score - a.score)
            setScores(ordenadas)
            registrar(ordenadas[0])
          }
          rafId = requestAnimationFrame(bucle)
        }
        bucle()
      } catch (e) {
        setEstado('Error: ' + e.message)
      }
    }

    iniciar()

    return () => {
      cancelado = true
      cancelAnimationFrame(rafId)
      stream?.getTracks().forEach((t) => t.stop())
      classifier?.close()
    }
  }, [])

  const top = scores[0]
  const identificado = top && top.score >= UMBRAL
  const total = Object.values(conteo).reduce((a, b) => a + b, 0)
  const categoriaActiva = identificado ? top.categoryName : 'none'
  const esError = estado.startsWith('Error')

  return (
    <div className="eco-app">
      {/* Cabecera Institucional / Operativa */}
      <header className="eco-header">
        <div className="eco-brand">
          <div className="eco-logo" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 19H4.815a1.83 1.83 0 0 1-1.57-.881 1.785 1.785 0 0 1-.004-1.784L7.196 9.5" />
              <path d="M11 19h8.203a1.83 1.83 0 0 0 1.556-.89 1.784 1.784 0 0 0 0-1.775l-1.226-2.12" />
              <path d="m14 16-3 3 3 3" />
              <path d="M8.293 13.596 7.196 9.5 3.1 10.598" />
              <path d="m9.344 5.811 1.093-1.892A1.83 1.83 0 0 1 11.985 3a1.784 1.784 0 0 1 1.546.888l3.943 6.843" />
              <path d="m13.378 9.633 4.096 1.098 1.097-4.096" />
            </svg>
          </div>
          <div>
            <h1>RecyIA</h1>
            <span className="eco-subtitle">Visión artificial para separación inteligente</span>
          </div>
        </div>

        <div
          className={`eco-status-badge ${
            estado === 'Listo' ? 'is-ready' : esError ? 'is-error' : 'is-loading'
          }`}
          role="status"
        >
          <span className="status-dot" />
          <span>{estado}</span>
        </div>
      </header>

      {/* Grid Principal: Visor (Izquierda) + Diagnóstico y Métricas (Derecha) */}
      <main className="eco-workspace">
        {/* Columna 1: Visor de Cámara y ROI */}
        <section className="eco-viewport-section" aria-label="Cámara de escaneo">
          <div className="viewport-frame">
            <video
              ref={videoRef}
              muted
              playsInline
              className="viewport-video"
            />
            {/* Máscara sutil fuera de la región de interés para guiar la vista */}
            <div className="viewport-vignette" />

            {/* Recuadro ROI exacto (left: 20%, top: 10%, width: 60%, height: 80%) */}
            <div className={`roi-target cat-${categoriaActiva}`}>
              <span className="roi-corner tl" />
              <span className="roi-corner tr" />
              <span className="roi-corner bl" />
              <span className="roi-corner br" />
              <div className="roi-label">
                {identificado
                  ? `${ETIQUETAS[top.categoryName] ?? top.categoryName} · ${Math.round(top.score * 100)}%`
                  : 'Centra el residuo aquí'}
              </div>
            </div>
          </div>

          <p className="viewport-hint">
            Coloca un solo objeto dentro del marco iluminado para evitar interferencias en la lectura.
          </p>
        </section>

        {/* Columna 2: Tarjeta de Resultado + Probabilidades + Estadísticas */}
        <aside className="eco-sidebar">
          {/* Tarjeta Hero: Resultado de Clasificación */}
          <section className={`eco-card result-card cat-${categoriaActiva}`} aria-live="polite">
            <div className="result-header">
              <span className="result-eyebrow">Detección actual</span>
              {identificado && (
                <span className="result-confidence">
                  Confianza: {(top.score * 100).toFixed(0)}%
                </span>
              )}
            </div>

            <h2 className="result-title">
              {identificado ? ETIQUETAS[top.categoryName] ?? top.categoryName : 'No identificado'}
            </h2>

            {identificado ? (
              <div className="result-guidance">
                <strong>Instrucción de depósito:</strong>
                <p>{RECOMENDACIONES[top.categoryName]}</p>
              </div>
            ) : (
              <p className="result-placeholder">
                Esperando un objeto con confianza superior al {Math.round(UMBRAL * 100)}%...
              </p>
            )}
          </section>

          {/* Tarjeta 2: Probabilidades en Tiempo Real */}
          <section className="eco-card">
            <div className="card-header">
              <h3>Probabilidades</h3>
              <span className="card-meta">Umbral: {UMBRAL.toFixed(2)}</span>
            </div>

            {scores.length === 0 ? (
              <p className="empty-state">Iniciando flujo de inferencia...</p>
            ) : (
              <ul className="prob-list">
                {scores.map((c) => {
                  const pct = Math.min(100, Math.max(0, c.score * 100))
                  const superaUmbral = c.score >= UMBRAL
                  return (
                    <li
                      key={c.categoryName}
                      className={`prob-item cat-${c.categoryName} ${superaUmbral ? 'is-active' : ''}`}
                    >
                      <div className="prob-meta">
                        <span className="prob-name">
                          <i className="cat-dot" />
                          {ETIQUETAS[c.categoryName] ?? c.categoryName}
                        </span>
                        <span className="prob-score">{c.score.toFixed(2)}</span>
                      </div>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>

          {/* Tarjeta 3: Estadísticas de Sesión */}
          <section className="eco-card">
            <div className="card-header">
              <div>
                <h3>Estadísticas</h3>
                <span className="card-meta">Conteo acumulado de sesión</span>
              </div>
              <div className="total-pill">
                <span>Total</span>
                <strong>{total}</strong>
              </div>
            </div>

            <ul className="stats-grid">
              {Object.keys(ETIQUETAS).map((k) => {
                const porcentaje = total ? Math.round((conteo[k] * 100) / total) : 0
                return (
                  <li key={k} className={`stat-tile cat-${k}`}>
                    <div className="stat-top">
                      <span className="stat-label">
                        <i className="cat-dot" />
                        {ETIQUETAS[k]}
                      </span>
                      <span className="stat-pct">{porcentaje}%</span>
                    </div>
                    <div className="stat-value">{conteo[k]}</div>
                    <div className="stat-mini-bar">
                      <div style={{ width: `${porcentaje}%` }} />
                    </div>
                  </li>
                )
              })}
            </ul>

            <button
              type="button"
              className="eco-btn-reset"
              onClick={() => setConteo(CONTEO_INICIAL)}
              disabled={total === 0}
            >
              Reiniciar estadísticas
            </button>
          </section>
        </aside>
      </main>
    </div>
  )
}