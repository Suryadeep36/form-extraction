import { useEffect, useState } from 'react';
import { useApi } from '../api.js';

export default function AuthImage({ src, alt, className, loading, ...props }) {
  const { apiFetch } = useApi();
  const [imgUrl, setImgUrl] = useState(null);

  useEffect(() => {
    let objectUrl = null;

    if (!src) return;

    if (src.startsWith('data:')) {
      setImgUrl(src);
      return;
    }

    apiFetch(src)
      .then(res => {
        if (!res.ok) throw new Error("Image fetch failed");
        return res.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        setImgUrl(objectUrl);
      })
      .catch(console.error);

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src]);

  if (!imgUrl) {
    return <div className={`bg-gray-200 animate-pulse ${className}`} />;
  }

  return <img src={imgUrl} alt={alt} className={className} loading={loading} {...props} />;
}
