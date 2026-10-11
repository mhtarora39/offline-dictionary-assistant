package com.offline.dictionary.voice

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale

class OfflineTextToSpeech(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isInitialized = false

    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    private val _currentUtteranceId = MutableStateFlow<String?>(null)
    val currentUtteranceId: StateFlow<String?> = _currentUtteranceId.asStateFlow()

    init {
        tts = TextToSpeech(context.applicationContext, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w("OfflineTextToSpeech", "US English language is not supported or missing data.")
            } else {
                isInitialized = true
                tts?.setSpeechRate(0.95f)
                tts?.setPitch(1.0f)
                setupProgressListener()
            }
        } else {
            Log.e("OfflineTextToSpeech", "TextToSpeech init failed with code $status")
        }
    }

    private fun setupProgressListener() {
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                _isSpeaking.value = true
                _currentUtteranceId.value = utteranceId
            }

            override fun onDone(utteranceId: String?) {
                _isSpeaking.value = false
                _currentUtteranceId.value = null
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                _isSpeaking.value = false
                _currentUtteranceId.value = null
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                _isSpeaking.value = false
                _currentUtteranceId.value = null
            }
        })
    }

    fun speakDefinition(
        headword: String,
        pos: String,
        definition: String,
        example: String? = null,
        utteranceId: String = "def_${headword}_${pos}"
    ) {
        val spokenPos = when (pos.lowercase()) {
            "n", "noun" -> "noun"
            "v", "verb" -> "verb"
            "adj", "adjective" -> "adjective"
            "adv", "adverb" -> "adverb"
            else -> pos
        }
        val builder = StringBuilder()
        builder.append(headword).append(". ")
        builder.append(spokenPos).append(". ")
        builder.append(definition).append(". ")
        if (!example.isNullOrBlank()) {
            builder.append("For example: ").append(example).append(". ")
        }
        speak(builder.toString(), utteranceId)
    }

    fun speak(text: String, utteranceId: String = "tts_${System.currentTimeMillis()}") {
        if (!isInitialized || text.isBlank()) return

        stop()

        val params = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
        }
        _currentUtteranceId.value = utteranceId
        _isSpeaking.value = true
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, utteranceId)
    }

    fun stop() {
        try {
            tts?.stop()
        } catch (_: Exception) {}
        _isSpeaking.value = false
        _currentUtteranceId.value = null
    }

    fun shutdown() {
        stop()
        try {
            tts?.shutdown()
        } catch (_: Exception) {}
        tts = null
        isInitialized = false
    }
}
