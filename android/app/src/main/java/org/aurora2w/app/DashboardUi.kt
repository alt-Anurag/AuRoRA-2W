package org.aurora2w.app

import android.app.Dialog
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.*
import android.graphics.drawable.*
import android.os.Bundle
import android.view.*
import android.widget.*
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.math.roundToInt

internal object AuroraColors {
    val background = Color.rgb(8,14,19)
    val panel = Color.rgb(17,27,35)
    val raised = Color.rgb(25,39,49)
    val edge = Color.rgb(39,57,68)
    val text = Color.rgb(237,245,244)
    val muted = Color.rgb(137,159,169)
    val mint = Color.rgb(92,231,222)
    val amber = Color.rgb(245,205,133)
    val blue = Color.rgb(126,193,236)
}

internal data class DashboardState(
    val cameraReady: Boolean = false, val paused: Boolean = false,
    val cameraStatus: String = "Starting camera", val modelLoaded: Boolean = false,
    val modelTitle: String = "Your camera is ready", val modelDetail: String = "Import a trained Aurora model to enable road, lane and pothole perception.",
    val importing: Boolean = false, val fps: Double? = null, val networkMs: Double? = null,
    val preprocessingMs: Double? = null,val postprocessingMs: Double? = null,val runtime: String = "Loading",
    val pipelineMs: Double? = null, val ageMs: Double? = null, val frameCount: Long = 0,
    val skipped: Long = 0, val objects: Int? = null, val potholes: Int? = null,
    val sensorStatus: String = "Waiting for camera and sensors", val roll: Float? = null,
    val rawRoll: Double? = null, val cameraAxes: String = "Camera axes pending",
    val imuUsed: Boolean = false, val recording: Boolean = false, val overlays: Boolean = true,
    val torchAvailable: Boolean = false,val torchOn: Boolean = false,val torchPending: Boolean = false
)

internal data class DashboardPreferences(val detection: Float = .35f,val mask: Float = .5f,
    val fps: Int = 15,val overlays: Boolean = true,val recordFrames: Boolean = false,
    val matchedFrames: Boolean = false)

internal data class DashboardActions(val pause: ()->Unit,val record: ()->Unit,val importModel: ()->Unit,
    val calibrate: ()->Unit,val controls: ()->Unit,val camera: ()->Unit,
    val motionReference: ()->Unit,val overlays: ()->Unit,val torch: ()->Unit)

/** Portrait cockpit with a full-frame camera and an independent phone-motion panel. */
internal class DashboardUi(private val context: Context,private val landscape: Boolean,
    private val actions: DashboardActions) {
    private val style=UiStyle(context)
    private val compact=context.resources.configuration.screenHeightDp < 700 && !landscape
    val root=LinearLayout(context).apply {orientation=LinearLayout.VERTICAL;setBackgroundColor(AuroraColors.background)}
    val cameraView=DashboardView(context)
    private val motionView=MotionView(context)
    private val live=style.label("CONNECTING",9f,AuroraColors.mint,true)
    private val cameraState=style.label("CAMERA ONLY",9f,AuroraColors.text,true)
    private val torch=style.iconButton("Camera torch unavailable","flash",actions.torch)
    private val maskStatus=style.label("Live camera · waiting for perception",10f,AuroraColors.amber)
    private val modelNote=style.label("No perception model loaded",11f,AuroraColors.text,true)
    private val importButton=style.button("Load","import",false,actions.importModel)
    private val fpsValue=style.label("—",21f,AuroraColors.text,true)
    private val inferenceValue=style.label("—",21f,AuroraColors.text,true)
    private val modelValue=style.label("WAITING",11f,AuroraColors.amber,true)
    private val motionStatus=style.label("Hold still briefly to start the bike",10f)
    private val acceleration=style.label("Acceleration deviation  —",10f)
    private val reference=style.button("Set reference","tilt",false,actions.motionReference)
    private val pause=style.button("Pause","pause",true,actions.pause)
    private val overlay=style.button("Overlay","layers",false,actions.overlays)
    private val record=style.button("Record","record",false,actions.record)
    init {
        ViewCompat.setOnApplyWindowInsetsListener(root) { v,insets ->
            val bars=insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val side=style.dp(if(landscape)16 else 14)
            v.setPadding(bars.left+side,bars.top+style.dp(4),bars.right+side,bars.bottom+style.dp(8))
            insets
        }
        root.addView(header(),LinearLayout.LayoutParams(-1,style.dp(if(landscape)46 else 54)))
        root.addView(style.space(8))
        if(landscape) buildLandscape() else buildPortrait()
    }
    private fun header(): View = LinearLayout(context).apply {
        gravity=Gravity.CENTER_VERTICAL
        addView(ImageView(context).apply {setImageResource(R.drawable.ic_aurora);importantForAccessibility=View.IMPORTANT_FOR_ACCESSIBILITY_NO},
            LinearLayout.LayoutParams(style.dp(32),style.dp(32)).apply {rightMargin=style.dp(9)})
        val title=LinearLayout(context).apply {orientation=LinearLayout.VERTICAL}
        val name=LinearLayout(context).apply {gravity=Gravity.CENTER_VERTICAL}
        name.addView(style.label("AURORA",20f,AuroraColors.text,true).apply {letterSpacing=.13f})
        name.addView(style.label("2W",10f,AuroraColors.mint,true),LinearLayout.LayoutParams(-2,-2).apply {leftMargin=style.dp(7)})
        title.addView(name);title.addView(style.label("R I D E   V I E W",8f))
        addView(title,LinearLayout.LayoutParams(0,-2,1f))
        if(landscape)addView(metrics(),LinearLayout.LayoutParams(style.dp(270),style.dp(44)).apply {rightMargin=style.dp(14)})
        addView(style.label("RESEARCH\nPREVIEW",8f).apply {gravity=Gravity.END},
            LinearLayout.LayoutParams(-2,-2).apply {rightMargin=style.dp(12)})
        addView(style.iconButton("Dashboard controls","settings",actions.controls))
    }
    private fun buildPortrait() {
        root.addView(hero(),LinearLayout.LayoutParams(-1,0,1f))
        root.addView(style.space(8))
        root.addView(metrics(),LinearLayout.LayoutParams(-1,style.dp(58)))
        root.addView(style.space(8))
        root.addView(motionPanel(),LinearLayout.LayoutParams(-1,style.dp(if(compact)200 else 218)))
        root.addView(style.space(10))
        root.addView(actionsRow(),LinearLayout.LayoutParams(-1,style.dp(48)))
    }
    private fun buildLandscape() {
        val row=LinearLayout(context)
        row.addView(hero(),LinearLayout.LayoutParams(0,-1,1f).apply {rightMargin=style.dp(12)})
        val rail=LinearLayout(context).apply {orientation=LinearLayout.VERTICAL}
        val details=LinearLayout(context).apply {orientation=LinearLayout.VERTICAL}
        details.addView(motionPanel(),LinearLayout.LayoutParams(-1,style.dp(206)))
        rail.addView(ScrollView(context).apply {addView(details);isFillViewport=false;isVerticalScrollBarEnabled=false},LinearLayout.LayoutParams(-1,0,1f))
        rail.addView(style.space(8))
        rail.addView(actionsRow(),LinearLayout.LayoutParams(-1,style.dp(48)))
        row.addView(rail,LinearLayout.LayoutParams(style.dp(310),-1))
        root.addView(row,LinearLayout.LayoutParams(-1,0,1f))
    }
    private fun hero(): View = FrameLayout(context).apply {
        background=style.round(Color.rgb(4,10,14),20,AuroraColors.edge);clipToOutline=true
        addView(cameraView,FrameLayout.LayoutParams(-1,-1))
        addView(View(context).apply {background=GradientDrawable(GradientDrawable.Orientation.TOP_BOTTOM,
            intArrayOf(Color.argb(100,4,10,14),Color.TRANSPARENT,Color.argb(95,4,10,14)))},FrameLayout.LayoutParams(-1,-1))
        val top=LinearLayout(context).apply {gravity=Gravity.CENTER_VERTICAL}
        live.apply {background=style.round(Color.argb(220,12,24,30),7);setPadding(style.dp(9),style.dp(7),style.dp(9),style.dp(7));letterSpacing=.05f}
        top.addView(live);top.addView(View(context),LinearLayout.LayoutParams(0,1,1f))
        top.addView(cameraState.apply {letterSpacing=.05f})
        top.addView(torch,LinearLayout.LayoutParams(style.dp(44),style.dp(44)).apply {leftMargin=style.dp(8)})
        addView(top,FrameLayout.LayoutParams(-1,-2,Gravity.TOP).apply {setMargins(style.dp(12),style.dp(12),style.dp(12),0)})
        addView(maskStatus.apply {setPadding(style.dp(8),style.dp(5),style.dp(8),style.dp(5))
            background=style.round(Color.argb(220,12,24,30),7)},FrameLayout.LayoutParams(-2,-2,Gravity.BOTTOM).apply {
                setMargins(style.dp(12),0,style.dp(12),style.dp(68))
            })
        val strip=LinearLayout(context).apply {
            gravity=Gravity.CENTER_VERTICAL;background=style.round(Color.argb(228,12,24,30),12,AuroraColors.edge)
            setPadding(style.dp(12),style.dp(3),style.dp(4),style.dp(3))
        }
        modelNote.maxLines=2
        strip.addView(modelNote,LinearLayout.LayoutParams(0,-2,1f).apply {rightMargin=style.dp(8)})
        importButton.textSize=11f;importButton.setPadding(style.dp(10),0,style.dp(10),0)
        strip.addView(importButton,LinearLayout.LayoutParams(-2,style.dp(44)))
        addView(strip,FrameLayout.LayoutParams(-1,-2,Gravity.BOTTOM).apply {setMargins(style.dp(10),0,style.dp(10),style.dp(10))})
    }
    private fun metrics(): View = LinearLayout(context).apply {
        gravity=Gravity.CENTER_VERTICAL;background=style.round(AuroraColors.panel,14,AuroraColors.edge)
        setPadding(style.dp(14),style.dp(if(landscape)3 else 8),style.dp(10),style.dp(if(landscape)3 else 8))
        val items=listOf(Triple("MODEL FPS",fpsValue,""),Triple("INFERENCE",inferenceValue,"ms"),Triple("PERCEPTION",modelValue,""))
        items.forEachIndexed { index,item ->
            val cell=LinearLayout(context).apply {orientation=LinearLayout.VERTICAL}
            cell.addView(style.label(item.first,8f,AuroraColors.muted,true).apply {letterSpacing=.06f})
            val line=LinearLayout(context).apply {gravity=Gravity.CENTER_VERTICAL}
            line.addView(item.second)
            if(item.third.isNotEmpty())line.addView(style.label(item.third,9f),LinearLayout.LayoutParams(-2,-2).apply {leftMargin=style.dp(4)})
            cell.addView(line,LinearLayout.LayoutParams(-1,style.dp(27)))
            addView(cell,LinearLayout.LayoutParams(0,-2,1f))
            if(index<2)addView(View(context).apply {setBackgroundColor(AuroraColors.edge)},
                LinearLayout.LayoutParams(style.dp(1),style.dp(30)).apply {rightMargin=style.dp(11)})
        }
    }
    private fun motionPanel(): View = LinearLayout(context).apply {
        orientation=LinearLayout.VERTICAL
        background=style.round(AuroraColors.panel,18,AuroraColors.edge)
        setPadding(style.dp(12),style.dp(5),style.dp(12),style.dp(9))
        val top=LinearLayout(context).apply {gravity=Gravity.CENTER_VERTICAL}
        top.addView(style.label("PHONE MOTION",9f,AuroraColors.text,true).apply {letterSpacing=.07f},
            LinearLayout.LayoutParams(0,-2,1f))
        reference.textSize=10f;reference.compoundDrawablePadding=style.dp(4);reference.setPadding(style.dp(9),0,style.dp(9),0)
        top.addView(reference,LinearLayout.LayoutParams(-2,style.dp(44)))
        addView(top,LinearLayout.LayoutParams(-1,style.dp(44)))
        addView(motionView,LinearLayout.LayoutParams(-1,0,1f))
        addView(motionStatus,LinearLayout.LayoutParams(-1,-2))
        addView(acceleration,LinearLayout.LayoutParams(-1,-2).apply {topMargin=style.dp(3)})
    }
    private fun actionsRow(): View = LinearLayout(context).apply {
        for((index,button) in listOf(pause,this@DashboardUi.overlay,record).withIndex()) {
            button.textSize=11f;button.setPadding(style.dp(8),0,style.dp(8),0);button.compoundDrawablePadding=style.dp(5)
            addView(button,LinearLayout.LayoutParams(0,-1,1f).apply {if(index<2)rightMargin=style.dp(7)})
        }
    }
    fun renderMaskStatus(value: String) = maskStatus.setIfChanged(value)
    fun clearMotionHistory() = motionView.clearHistory()
    fun renderMotion(value: MotionReading,nowNs: Long) {
        motionView.submit(value,nowNs)
        motionStatus.setIfChanged(value.status)
        motionStatus.setTextColor(if(value.rollDegrees != null)AuroraColors.mint else AuroraColors.muted)
        acceleration.setIfChanged(value.accelerationDeviation?.let {String.format(java.util.Locale.US,"Acceleration deviation  %.2f m/s²",it)} ?: "Acceleration deviation  —")
        reference.setIfChanged(if(value.referenced)"Recenter" else "Hold still")
    }
    fun render(state: DashboardState) {
        torch.isEnabled=state.torchAvailable && !state.torchPending
        torch.alpha=if(state.torchAvailable)1f else .35f
        torch.contentDescription=when { !state.torchAvailable->"Torch unavailable on this camera";state.torchPending->"Changing torch";state.torchOn->"Turn torch off";else->"Turn torch on" }
        (torch as FrameLayout).getChildAt(0).let { icon -> (icon as ImageView).setImageDrawable(AuroraIcon("flash",if(state.torchOn)AuroraColors.amber else AuroraColors.text,style.dp(20))) }

        live.setIfChanged(when {state.paused->"Ⅱ  PAUSED";state.cameraReady->"●  LIVE CAMERA";else->"○  CONNECTING"})
        live.setTextColor(if(state.paused)AuroraColors.amber else AuroraColors.mint)
        cameraState.setIfChanged(if(state.modelLoaded)"PERCEPTION ON" else "CAMERA ONLY")
        modelNote.setIfChanged(when {
            state.importing->"Checking model bundle…"
            !state.cameraReady->state.cameraStatus
            state.modelLoaded->state.modelTitle
            else->"No perception model loaded"
        })
        importButton.setIfChanged(when {state.importing->"Checking";!state.cameraReady->"Open camera";state.modelLoaded->"Model";else->"Load"})
        importButton.isEnabled=!state.importing
        importButton.setOnClickListener {if(!state.cameraReady)actions.camera() else if(state.modelLoaded)actions.controls() else actions.importModel()}
        fpsValue.setIfChanged(if(state.paused)"—" else state.fps?.let {"%.1f".format(java.util.Locale.US,it)} ?: "—")
        inferenceValue.setIfChanged(if(state.paused)"—" else state.networkMs?.let {"%.0f".format(java.util.Locale.US,it)} ?: "—")
        modelValue.setIfChanged(if(state.modelLoaded)"ACTIVE" else "WAITING")
        modelValue.setTextColor(if(state.modelLoaded)AuroraColors.mint else AuroraColors.amber)
        pause.setIfChanged(if(state.paused)"Resume" else "Pause")
        pause.setCompoundDrawablesWithIntrinsicBounds(style.drawable(if(state.paused)"play" else "pause",AuroraColors.background),null,null,null)
        overlay.setTextColor(if(state.overlays)AuroraColors.mint else AuroraColors.muted)
        overlay.setCompoundDrawablesWithIntrinsicBounds(style.drawable("layers",if(state.overlays)AuroraColors.mint else AuroraColors.muted),null,null,null)
        overlay.contentDescription=if(state.overlays)"Road and lane overlays on" else "Road and lane overlays off"
        record.setIfChanged(if(state.recording)"Stop" else "Record")
        record.setTextColor(if(state.recording)AuroraColors.amber else AuroraColors.text)
        record.setCompoundDrawablesWithIntrinsicBounds(style.drawable(if(state.recording)"stop" else "record",if(state.recording)AuroraColors.amber else AuroraColors.text),null,null,null)
    }
}


internal fun TextView.setIfChanged(value: String) { if (text.toString() != value) text = value }

/** Small native primitives keep the main view and control sheet visually consistent. */
internal class UiStyle(private val context: Context) {
    fun dp(value: Int) = (value*context.resources.displayMetrics.density).roundToInt()
    fun space(height: Int): View = View(context).apply { layoutParams = LinearLayout.LayoutParams(1,dp(height)) }
    fun label(value: String,size: Float = 12f,color: Int = AuroraColors.muted,bold: Boolean = false): TextView = TextView(context).apply {
        text = value; textSize = size; setTextColor(color); includeFontPadding = false
        typeface = Typeface.create(if (bold) "sans-serif-medium" else "sans-serif",Typeface.NORMAL)
        setLineSpacing(dp(2).toFloat(),1f)
    }
    fun round(color: Int,radius: Int,stroke: Int? = null): GradientDrawable = GradientDrawable().apply {
        setColor(color); cornerRadius = dp(radius).toFloat(); if (stroke != null) setStroke(dp(1),stroke)
    }
    fun drawable(kind: String,color: Int = AuroraColors.text): Drawable = AuroraIcon(kind,color,dp(18)).apply { setBounds(0,0,dp(18),dp(18)) }
    fun icon(kind: String,color: Int,size: Int): View = ImageView(context).apply { setImageDrawable(AuroraIcon(kind,color,dp(size))) }
    fun button(value: String,kind: String,primary: Boolean = false,action: ()->Unit): TextView = label(value,13f,if (primary) AuroraColors.background else AuroraColors.text,true).apply {
        gravity = Gravity.CENTER; minHeight = dp(44); isClickable = true; isFocusable = true
        background = RippleDrawable(ColorStateList.valueOf(Color.argb(50,255,255,255)),round(if (primary) AuroraColors.mint else AuroraColors.raised,13),null)
        setPadding(dp(16),0,dp(16),0); compoundDrawablePadding = dp(8)
        setCompoundDrawablesWithIntrinsicBounds(drawable(kind,if (primary) AuroraColors.background else AuroraColors.text),null,null,null)
        setOnClickListener { action() }
    }
    fun iconButton(label: String,kind: String,action: ()->Unit): View = FrameLayout(context).apply {
        layoutParams = LinearLayout.LayoutParams(dp(44),dp(44)); contentDescription = label; isClickable = true; isFocusable = true
        background = RippleDrawable(ColorStateList.valueOf(Color.argb(60,255,255,255)),round(AuroraColors.raised,13),null)
        addView(icon(kind,AuroraColors.text,20),FrameLayout.LayoutParams(dp(20),dp(20),Gravity.CENTER))
        setOnClickListener { action() }
    }
}

internal class AuroraIcon(private val kind: String,private val tint: Int,private val size: Int): Drawable() {
    private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 1.8f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND }
    override fun draw(canvas: Canvas) {
        canvas.save(); canvas.translate(bounds.left.toFloat(),bounds.top.toFloat()); canvas.scale(bounds.width()/24f,bounds.height()/24f); p.color = tint
        fun line(a: Float,b: Float,c: Float,d: Float) = canvas.drawLine(a,b,c,d,p)
        when (kind) {
            "flash" -> { val bolt=Path();bolt.moveTo(14f,2f);bolt.lineTo(5f,13f);bolt.lineTo(11f,13f);bolt.lineTo(10f,22f);bolt.lineTo(20f,10f);bolt.lineTo(14f,10f);bolt.close();canvas.drawPath(bolt,p) }
            "layers" -> { line(4f,7f,12f,3f);line(12f,3f,20f,7f);line(20f,7f,12f,11f);line(12f,11f,4f,7f);line(4f,12f,12f,16f);line(12f,16f,20f,12f);line(4f,17f,12f,21f);line(12f,21f,20f,17f) }
            "settings" -> { line(4f,6f,20f,6f);line(4f,12f,20f,12f);line(4f,18f,20f,18f);p.style=Paint.Style.FILL;canvas.drawCircle(9f,6f,3f,p);canvas.drawCircle(16f,12f,3f,p);canvas.drawCircle(8f,18f,3f,p) }
            "pause" -> { line(8f,5f,8f,19f);line(16f,5f,16f,19f) }
            "play" -> { val path=Path();path.moveTo(7f,4f);path.lineTo(20f,12f);path.lineTo(7f,20f);path.close();canvas.drawPath(path,p) }
            "record" -> canvas.drawCircle(12f,12f,7f,p)
            "stop" -> canvas.drawRoundRect(5f,5f,19f,19f,2f,2f,p)
            "import" -> { line(12f,3f,12f,15f);line(7f,10f,12f,15f);line(12f,15f,17f,10f);line(4f,16f,4f,21f);line(4f,21f,20f,21f);line(20f,21f,20f,16f) }
            "chevron" -> { line(9f,5f,16f,12f);line(16f,12f,9f,19f) }
            "close" -> { line(6f,6f,18f,18f);line(18f,6f,6f,18f) }
            "tilt" -> { canvas.save();canvas.rotate(-18f,12f,12f);canvas.drawRoundRect(6f,3f,18f,21f,3f,3f,p);line(10f,17f,14f,17f);canvas.restore() }
            "share" -> { line(12f,17f,12f,3f);line(7f,8f,12f,3f);line(17f,8f,12f,3f);line(4f,13f,4f,21f);line(4f,21f,20f,21f);line(20f,21f,20f,13f) }
        }
        p.style = Paint.Style.STROKE; canvas.restore()
    }
    override fun getIntrinsicWidth() = size
    override fun getIntrinsicHeight() = size
    override fun setAlpha(alpha: Int) { p.alpha = alpha }
    override fun setColorFilter(filter: ColorFilter?) { p.colorFilter = filter }
    @Deprecated("Deprecated in Java") override fun getOpacity() = PixelFormat.TRANSLUCENT
}

internal class DashboardControls(context: Context,private val state: DashboardState,
    private var preferences: DashboardPreferences,private val onChange: (DashboardPreferences)->Unit,
    private val onImport: ()->Unit,private val onCalibrate: ()->Unit,private val onClear: ()->Unit,
    private val onShare: ()->Unit,private val onCamera: ()->Unit): Dialog(context) {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val s = UiStyle(context)
        val page = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL; background = s.round(AuroraColors.panel,24); setPadding(s.dp(20),s.dp(18),s.dp(20),s.dp(20)) }
        val header = LinearLayout(context).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(s.label("Dashboard controls",21f,AuroraColors.text,true),LinearLayout.LayoutParams(0,-2,1f))
        header.addView(s.iconButton("Close controls","close") { dismiss() })
        page.addView(header);page.addView(s.space(14))
        val scroll = ScrollView(context)
        val column = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        fun section(title: String) { column.addView(s.label(title,10f,AuroraColors.mint,true).apply { letterSpacing=.12f },LinearLayout.LayoutParams(-1,-2).apply { topMargin=s.dp(18);bottomMargin=s.dp(10) }) }
        fun button(label: String,icon: String,action: ()->Unit) { column.addView(s.button(label,icon,false,action),LinearLayout.LayoutParams(-1,s.dp(48)).apply { topMargin=s.dp(8) }) }
        fun slider(title: String,initial: Int,min: Int,max: Int,suffix: String,changed: (Int)->Unit) {
            val label=s.label("$title  $initial$suffix",13f,AuroraColors.text)
            column.addView(label,LinearLayout.LayoutParams(-1,-2).apply { topMargin=s.dp(10) })
            column.addView(SeekBar(context).apply {
                this.max=max-min;progress=initial-min;progressTintList=ColorStateList.valueOf(AuroraColors.mint);thumbTintList=ColorStateList.valueOf(AuroraColors.mint)
                setPadding(0,s.dp(4),0,s.dp(4))
                setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(bar: SeekBar?,value: Int,user: Boolean) {label.setIfChanged("$title  ${value+min}$suffix");changed(value+min)}
                    override fun onStartTrackingTouch(bar: SeekBar?)=Unit
                    override fun onStopTrackingTouch(bar: SeekBar?)=Unit
                })
            },LinearLayout.LayoutParams(-1,s.dp(38)))
        }
        fun toggle(title: String,checked: Boolean,changed: (Boolean)->Unit) {
            column.addView(Switch(context).apply {text=title;textSize=13f;setTextColor(AuroraColors.text);isChecked=checked;minHeight=s.dp(46);setOnCheckedChangeListener { _,value->changed(value) }})
        }
        section("MODEL")
        column.addView(s.label(if(state.modelLoaded)state.modelTitle else "No trained model loaded",16f,AuroraColors.text,true))
        column.addView(s.label(if(state.modelLoaded)state.modelDetail else "Import an Aurora or prepared YOLOP bundle. The app verifies its model, metadata and runtime compatibility.",12f),LinearLayout.LayoutParams(-1,-2).apply {topMargin=s.dp(6)})
        button("Import model bundle","import") {dismiss();onImport()}
        section("PERCEPTION")
        slider("Object confidence",(preferences.detection*100).roundToInt(),5,95,"%") {preferences=preferences.copy(detection=it/100f);onChange(preferences)}
        slider("Mask confidence",(preferences.mask*100).roundToInt(),5,95,"%") {preferences=preferences.copy(mask=it/100f);onChange(preferences)}
        slider("Maximum model rate",preferences.fps,5,30," fps") {preferences=preferences.copy(fps=it);onChange(preferences)}
        column.addView(s.label("This caps model work to control heat. Raising it cannot exceed the phone's inference speed. Live camera preview runs independently.",12f))
        toggle("Match camera frames to masks",preferences.matchedFrames) {preferences=preferences.copy(matchedFrames=it);onChange(preferences)}
        column.addView(s.label("Off: smooth live preview with delayed masks drawn at model resolution. Masks hide after 0.5 s or a large turn; movement can still cause small offsets. On: exact frame alignment at the model's speed.",12f))
        toggle("Road and lane overlay",preferences.overlays) {preferences=preferences.copy(overlays=it);onChange(preferences)}
        section("MOUNT & SENSORS")
        column.addView(s.label(state.sensorStatus,13f,AuroraColors.text))
        column.addView(s.label("Hold level and still, with the visible horizon level. Do not zero a tilted mount. Each camera orientation keeps its own reference; this does not correct arbitrary pitch, yaw or motion blur.",12f),LinearLayout.LayoutParams(-1,-2).apply {topMargin=s.dp(6)})
        column.addView(s.label("Camera roll   ${state.rawRoll?.let { "%+.1f°".format(Math.toDegrees(it)) } ?: "Unavailable"}\nModel roll   ${state.roll?.let { "%+.1f°".format(Math.toDegrees(it.toDouble())) } ?: "Disabled until calibrated"}\n${state.cameraAxes}",12f),LinearLayout.LayoutParams(-1,-2).apply {topMargin=s.dp(10)})
        button("Calibrate level mount","tilt") {dismiss();onCalibrate()}
        button("Clear level references","close") {onClear()}
        section("TRACE RECORDING")
        toggle("Include camera frames",preferences.recordFrames) {preferences=preferences.copy(recordFrames=it);onChange(preferences)}
        column.addView(s.label("Optional JPEGs: at most 2 fps for the first 60 seconds. Camera timestamps and IMU events stay in app storage until you share them.",12f))
        button("Share last trace","share") {dismiss();onShare()}
        section("DIAGNOSTICS")
        column.addView(s.label("Pipeline   ${state.pipelineMs?.let { "%.1f ms".format(it) } ?: "—"}\nCapture → UI   ${state.ageMs?.let { "%.0f ms".format(it) } ?: "Clock unavailable"}\nFrames processed   ${state.frameCount}\nFrames skipped   ${state.skipped}\nPreprocessing   ${state.preprocessingMs?.let { "%.1f ms".format(it) } ?: "—"}\nMask rendering   ${state.postprocessingMs?.let { "%.1f ms".format(it) } ?: "—"}\nRuntime   ${state.runtime}",12f))
        button("Restart camera","play") {dismiss();onCamera()}
        column.addView(s.label("Research preview · device accuracy and sustained performance require validation.",11f),LinearLayout.LayoutParams(-1,-2).apply {topMargin=s.dp(18);bottomMargin=s.dp(8)})
        scroll.addView(column);page.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        setContentView(page)
        window?.apply {
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND);setDimAmount(.6f)
            val landscape=context.resources.configuration.screenWidthDp>context.resources.configuration.screenHeightDp
            setGravity(if(landscape)Gravity.END else Gravity.BOTTOM)
            setLayout(if(landscape)s.dp(380) else -1,(context.resources.displayMetrics.heightPixels*.91).roundToInt())
        }
    }
}
