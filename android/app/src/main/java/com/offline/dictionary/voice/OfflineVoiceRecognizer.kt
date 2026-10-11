package com.offline.dictionary.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

class OfflineVoiceRecognizer(private val context: Context) {

    private var speechRecognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isStopping: Boolean = false
    private var stateCallback: ((Boolean) -> Unit)? = null

    var isListening: Boolean = false
        private set

    fun startListening(
        onListeningStateChanged: (Boolean) -> Unit,
        onPartialResult: (String) -> Unit,
        onFinalResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            onListeningStateChanged(false)
            onError("Speech recognition is not available on this device.")
            return
        }

        stopListening()
        isStopping = false
        stateCallback = onListeningStateChanged

        mainHandler.post {
            try {
                val recognizer = SpeechRecognizer.createSpeechRecognizer(context)
                speechRecognizer = recognizer

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
                    putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
                }

                recognizer.setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        isListening = true
                        mainHandler.post { stateCallback?.invoke(true) }
                    }

                    override fun onBeginningOfSpeech() {}
                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}

                    override fun onEndOfSpeech() {
                        isListening = false
                        mainHandler.post { stateCallback?.invoke(false) }
                    }

                    override fun onError(error: Int) {
                        isListening = false
                        mainHandler.post { stateCallback?.invoke(false) }

                        if (isStopping) return

                        val msg = when (error) {
                            SpeechRecognizer.ERROR_NO_MATCH -> "No speech recognized. Tap mic to speak."
                            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout. Tap mic to speak."
                            SpeechRecognizer.ERROR_AUDIO -> "Audio recording error. Check microphone."
                            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission required."
                            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Speech service is busy. Please try again."
                            SpeechRecognizer.ERROR_CLIENT -> null // Ignored on cancel or teardown
                            SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Speech service network error."
                            else -> "Speech recognition error ($error)"
                        }
                        if (msg != null) {
                            mainHandler.post { onError(msg) }
                        }
                    }

                    override fun onResults(results: Bundle?) {
                        isListening = false
                        mainHandler.post { stateCallback?.invoke(false) }

                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val text = matches?.firstOrNull()?.trim() ?: ""
                        if (text.isNotEmpty()) {
                            mainHandler.post { onFinalResult(text) }
                        }
                    }

                    override fun onPartialResults(partialResults: Bundle?) {
                        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val text = matches?.firstOrNull()?.trim() ?: ""
                        if (text.isNotEmpty()) {
                            mainHandler.post { onPartialResult(text) }
                        }
                    }

                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })

                recognizer.startListening(intent)
                isListening = true
                onListeningStateChanged(true)
            } catch (e: Exception) {
                isListening = false
                onListeningStateChanged(false)
                onError("Failed to start speech recognizer: ${e.message}")
            }
        }
    }

    fun stopListening() {
        isStopping = true
        isListening = false
        val cb = stateCallback
        mainHandler.post {
            cb?.invoke(false)
            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.cancel()
                speechRecognizer?.destroy()
            } catch (_: Exception) {}
            speechRecognizer = null
        }
    }
}
