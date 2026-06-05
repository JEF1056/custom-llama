/**
 * PDF → base64 PNG converter using pdfjs-dist Web Worker.
 * Cap: 20 pages. Scale: 1.5x.
 */

const MAX_PAGES = 20
const SCALE = 1.5

// pdfjs worker loaded via CDN or bundled — Vite will resolve the static import
async function getPdfJs() {
  const pdfjs = await import('pdfjs-dist')
  // Configure worker — use the bundled worker via URL import trick
  if (!pdfjs.GlobalWorkerOptions.workerSrc) {
    // Vite handles the ?url suffix for static asset imports
    const workerUrl = new URL('pdfjs-dist/build/pdf.worker.mjs', import.meta.url).href
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
  }
  return pdfjs
}

/**
 * Convert a PDF File to an array of base64-encoded PNG strings (one per page).
 * Returns at most MAX_PAGES pages.
 */
export async function pdfToImages(file: File): Promise<{ pages: string[]; pageCount: number }> {
  const pdfjs = await getPdfJs()

  const arrayBuffer = await file.arrayBuffer()
  const loadingTask = pdfjs.getDocument({ data: arrayBuffer })
  const doc = await loadingTask.promise

  const totalPages = doc.numPages
  const pagesToRender = Math.min(totalPages, MAX_PAGES)
  const pages: string[] = []

  for (let pageNum = 1; pageNum <= pagesToRender; pageNum++) {
    const page = await doc.getPage(pageNum)
    const viewport = page.getViewport({ scale: SCALE })

    // Use OffscreenCanvas when available (Workers + modern browsers)
    let canvas: HTMLCanvasElement | OffscreenCanvas
    let ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

    if (typeof OffscreenCanvas !== 'undefined') {
      canvas = new OffscreenCanvas(viewport.width, viewport.height)
      ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D
    } else {
      canvas = document.createElement('canvas')
      canvas.width = viewport.width
      canvas.height = viewport.height
      ctx = canvas.getContext('2d') as CanvasRenderingContext2D
    }

    await page.render({ canvasContext: ctx as CanvasRenderingContext2D, viewport }).promise

    let blob: Blob
    if (canvas instanceof OffscreenCanvas) {
      blob = await canvas.convertToBlob({ type: 'image/png' })
    } else {
      blob = await new Promise<Blob>((resolve, reject) => {
        ;(canvas as HTMLCanvasElement).toBlob((b) => {
          if (b) resolve(b)
          else reject(new Error('Canvas toBlob failed'))
        }, 'image/png')
      })
    }

    const base64 = await blobToBase64(blob)
    pages.push(base64)

    page.cleanup()
  }

  await doc.destroy()
  return { pages, pageCount: totalPages }
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Strip the data:image/png;base64, prefix — caller adds it back
      resolve(result.split(',')[1] ?? result)
    }
    reader.onerror = () => reject(new Error('FileReader error'))
    reader.readAsDataURL(blob)
  })
}
