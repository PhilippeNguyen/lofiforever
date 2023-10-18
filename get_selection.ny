;nyquist plug-in
;version 4
;type analyze
;name "Set Selection Times as Labels"

(setq start (get '*selection* 'start))
(setq end (get '*selection* 'end))

(list (list start end (format nil "Start: ~a, End: ~a" start end)))
