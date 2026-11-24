const multer = require('multer');
const path = require('path');
const fs = require('fs');

const uploadDir = path.join(__dirname, '../../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const sanitizedName = file.originalname.replace(/\s+/g, '_');
    cb(null, `${timestamp}-${sanitizedName}`);
  },
});

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  const originalName = file.originalname.toLowerCase();

  if (originalName.endsWith('.nii.gz') || ext === '.nii') {
    cb(null, true);
  } else {
    cb(new Error('Only NIfTI medical imaging files (.nii or .nii.gz) are supported.'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit for high-res 3D DCE-MRI
  },
});

module.exports = upload;
