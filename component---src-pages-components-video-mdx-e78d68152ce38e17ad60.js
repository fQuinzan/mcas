(window.webpackJsonp=window.webpackJsonp||[]).push([[38],{"013z":function(e,t,a){"use strict";var n=a("q1tI"),r=a.n(n),b=a("NmYn"),o=a.n(b),l=a("Wbzz"),i=a("Xrax"),c=a("k4MR"),d=a("TSYQ"),s=a.n(d),m=a("QH2O"),p=a.n(m),u=a("qKvR"),g=function(e){var t,a=e.title,n=e.theme,r=e.tabs,b=void 0===r?[]:r;return Object(u.b)("div",{className:s()(p.a.pageHeader,(t={},t[p.a.withTabs]=b.length,t[p.a.darkMode]="dark"===n,t))},Object(u.b)("div",{className:"bx--grid"},Object(u.b)("div",{className:"bx--row"},Object(u.b)("div",{className:"bx--col-lg-12"},Object(u.b)("h1",{id:"page-title",className:p.a.text},a)))))},j=a("BAC9"),O=function(e){var t=e.relativePagePath,a=e.repository,n=Object(l.useStaticQuery)("1364590287").site.siteMetadata.repository,r=a||n,b=r.baseUrl,o=r.subDirectory,i=b+"/edit/"+r.branch+o+"/src/pages"+t;return b?Object(u.b)("div",{className:"bx--row "+j.row},Object(u.b)("div",{className:"bx--col"},Object(u.b)("a",{className:j.link,href:i},"Edit this page on GitHub"))):null},h=a("FCXl"),N=a("dI71"),v=a("I8xM"),x=function(e){function t(){return e.apply(this,arguments)||this}return Object(N.a)(t,e),t.prototype.render=function(){var e=this.props,t=e.title,a=e.tabs,n=e.slug,r=n.split("/").filter(Boolean).slice(-1)[0],b=a.map((function(e){var t,a=o()(e,{lower:!0,strict:!0}),b=a===r,i=new RegExp(r+"/?(#.*)?$"),c=n.replace(i,a);return Object(u.b)("li",{key:e,className:s()((t={},t[v.selectedItem]=b,t),v.listItem)},Object(u.b)(l.Link,{className:v.link,to:""+c},e))}));return Object(u.b)("div",{className:v.tabsContainer},Object(u.b)("div",{className:"bx--grid"},Object(u.b)("div",{className:"bx--row"},Object(u.b)("div",{className:"bx--col-lg-12 bx--col-no-gutter"},Object(u.b)("nav",{"aria-label":t},Object(u.b)("ul",{className:v.list},b))))))},t}(r.a.Component),y=a("MjG9"),T=a("CzIb");t.a=function(e){var t=e.pageContext,a=e.children,n=e.location,r=e.Title,b=t.frontmatter,d=void 0===b?{}:b,s=t.relativePagePath,m=t.titleType,p=d.tabs,j=d.title,N=d.theme,v=d.description,f=d.keywords,k=Object(T.a)().interiorTheme,w=Object(l.useStaticQuery)("2456312558").site.pathPrefix,P=w?n.pathname.replace(w,""):n.pathname,V=p?P.split("/").filter(Boolean).slice(-1)[0]||o()(p[0],{lower:!0}):"",C=N||k;return Object(u.b)(c.a,{tabs:p,homepage:!1,theme:C,pageTitle:j,pageDescription:v,pageKeywords:f,titleType:m},Object(u.b)(g,{title:r?Object(u.b)(r,null):j,label:"label",tabs:p,theme:C}),p&&Object(u.b)(x,{title:j,slug:P,tabs:p,currentTab:V}),Object(u.b)(y.a,{padded:!0},a,Object(u.b)(O,{relativePagePath:s})),Object(u.b)(h.a,{pageContext:t,location:n,slug:P,tabs:p,currentTab:V}),Object(u.b)(i.a,null))}},BAC9:function(e,t,a){e.exports={bxTextTruncateEnd:"EditLink-module--bx--text-truncate--end--2pqje",bxTextTruncateFront:"EditLink-module--bx--text-truncate--front--3_lIE",link:"EditLink-module--link--1qzW3",row:"EditLink-module--row--1B9Gk"}},I8xM:function(e,t,a){e.exports={bxTextTruncateEnd:"PageTabs-module--bx--text-truncate--end--267NA",bxTextTruncateFront:"PageTabs-module--bx--text-truncate--front--3xEQF",tabsContainer:"PageTabs-module--tabs-container--8N4k0",list:"PageTabs-module--list--3eFQc",listItem:"PageTabs-module--list-item--nUmtD",link:"PageTabs-module--link--1mDJ1",selectedItem:"PageTabs-module--selected-item--YPVr3"}},QH2O:function(e,t,a){e.exports={bxTextTruncateEnd:"PageHeader-module--bx--text-truncate--end--mZWeX",bxTextTruncateFront:"PageHeader-module--bx--text-truncate--front--3zvrI",pageHeader:"PageHeader-module--page-header--3hIan",darkMode:"PageHeader-module--dark-mode--hBrwL",withTabs:"PageHeader-module--with-tabs--3nKxA",text:"PageHeader-module--text--o9LFq"}},yylr:function(e,t,a){"use strict";a.r(t),a.d(t,"_frontmatter",(function(){return l})),a.d(t,"default",(function(){return u}));var n=a("wx14"),r=a("zLVn"),b=(a("q1tI"),a("7ljp")),o=a("013z"),l=(a("qKvR"),{}),i=function(e){return function(t){return console.warn("Component "+e+" was not imported, exported, or provided by MDXProvider as global scope"),Object(b.b)("div",t)}},c=i("PageDescription"),d=i("Title"),s=i("Video"),m={_frontmatter:l},p=o.a;function u(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(b.b)(p,Object(n.a)({},m,a,{components:t,mdxType:"MDXLayout"}),Object(b.b)(c,{mdxType:"PageDescription"},Object(b.b)("p",null,"The ",Object(b.b)("inlineCode",{parentName:"p"},"<Video>")," component can render a Vimeo player or a html video player.")),Object(b.b)("h2",null,"Example"),Object(b.b)(d,{mdxType:"Title"},"Vimeo"),Object(b.b)(s,{title:"Carbon homepage video",vimeoId:"359578263",mdxType:"Video"}),Object(b.b)(d,{mdxType:"Title"},"Video"),Object(b.b)(s,{src:"/videos/hero-video.mp4",poster:"/images/poster.png",mdxType:"Video"},Object(b.b)("track",{kind:"captions",default:!0,src:"/videos/vtt/hero-video.vtt",srcLang:"en"})),Object(b.b)("h2",null,"Code"),Object(b.b)(d,{mdxType:"Title"},"Vimeo"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Video/Video.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Video",path:"components/Video/Video.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Video"},'<Video title="Carbon homepage video" vimeoId="322021187" />\n')),Object(b.b)(d,{mdxType:"Title"},"Video"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Video/Video.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Video",path:"components/Video/Video.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Video"},'<Video src="/videos/hero-video.mp4" poster="/images/poster.png">\n  <track kind="captions" default src="/videos/vtt/hero-video.vtt" srcLang="en" />\n</Video>\n')),Object(b.b)("h3",null,"Props"),Object(b.b)("table",null,Object(b.b)("thead",{parentName:"table"},Object(b.b)("tr",{parentName:"thead"},Object(b.b)("th",{parentName:"tr",align:null},"property"),Object(b.b)("th",{parentName:"tr",align:null},"propType"),Object(b.b)("th",{parentName:"tr",align:null},"required"),Object(b.b)("th",{parentName:"tr",align:null},"default"),Object(b.b)("th",{parentName:"tr",align:null},"description"))),Object(b.b)("tbody",{parentName:"table"},Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"vimeoId"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"To find your ",Object(b.b)("inlineCode",{parentName:"td"},"vimeoId"),", go to the Vimeo page and find the video you want to put on your website. Once it is loaded, look at the URL and look for the numbers that come after the slash (/).")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"src"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Use the html ",Object(b.b)("inlineCode",{parentName:"td"},"<video>")," player with a local ",Object(b.b)("inlineCode",{parentName:"td"},".mp4")," video")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"title"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Vimeo title")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"poster"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Provides an image to show before the video loads, only works with ",Object(b.b)("inlineCode",{parentName:"td"},"src"))),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"children"),Object(b.b)("td",{parentName:"tr",align:null},Object(b.b)("a",{parentName:"td",href:"https://developer.mozilla.org/en-US/docs/Web/HTML/Element/track"},Object(b.b)("inlineCode",{parentName:"a"},"<track>"))),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},Object(b.b)("em",{parentName:"td"},"non-vimeo only")," – Provide ",Object(b.b)("inlineCode",{parentName:"td"},".vtt")," file in your static directory to make your videos more accessible. Then add a track element with a src pointing to it Check out ",Object(b.b)("a",{parentName:"td",href:"https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API#Tutorial_on_how_to_write_a_WebVTT_file"},"this simple tutorial")," for getting started with writing vtt files.")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"autoPlay"),Object(b.b)("td",{parentName:"tr",align:null},"boolean"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Whether or not the video should autoplay.")))))}u.isMDXComponent=!0}}]);
//# sourceMappingURL=component---src-pages-components-video-mdx-e78d68152ce38e17ad60.js.map