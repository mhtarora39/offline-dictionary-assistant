package com.offline.dictionary

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.offline.dictionary.ui.DictionaryScreen
import com.offline.dictionary.ui.DictionaryViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: DictionaryViewModel by viewModels()

    private val speechDialogLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val matches = result.data?.getStringArrayListExtra(android.speech.RecognizerIntent.EXTRA_RESULTS)
            val spokenText = matches?.firstOrNull()?.trim()
            if (!spokenText.isNullOrEmpty()) {
                viewModel.onVoiceResult(spokenText)
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.toggleVoiceListening(
                onFallbackLaunchIntent = { intent -> speechDialogLauncher.launch(intent) }
            )
        } else {
            viewModel.onQueryChanged(viewModel.uiState.value.query)
        }
    }

    private fun handleMicClick() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            viewModel.toggleVoiceListening(
                onFallbackLaunchIntent = { intent -> speechDialogLauncher.launch(intent) }
            )
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DictionaryScreen(
                viewModel = viewModel,
                onMicClick = { handleMicClick() }
            )
        }
    }
}
